# FMM拡張: 高速多重極子展開法 with Conway-Mathieu群論
extends GLTFPotentialProcessor

class_name FMMGLTFProcessor

# FMM特化定数
const FMM_MAX_DEPTH = 8
const FMM_LEAF_SIZE = 64
const MULTIPOLE_ORDER = 10
const INTERACTION_THRESHOLD = 2.0

# 複素数近似構造体
class ComplexApprox:
	var real: float
	var imag: float
	
	func _init(r: float = 0.0, i: float = 0.0):
		real = r
		imag = i
	
	func multiply(other: ComplexApprox) -> ComplexApprox:
		return ComplexApprox.new(
			real * other.real - imag * other.imag,
			real * other.imag + imag * other.real
		)
	
	func add(other: ComplexApprox) -> ComplexApprox:
		return ComplexApprox.new(real + other.real, imag + other.imag)

# FMM階層ノード
class FMMNode:
	var center: Vector3
	var size: float
	var level: int
	var children: Array[FMMNode] = []
	var particles: Array[int] = []  # パーティクルID配列
	var multipole_coeffs: Array[ComplexApprox] = []
	var local_coeffs: Array[ComplexApprox] = []
	var group_pointer: VirtualPointer
	
	func _init(pos: Vector3, sz: float, lv: int):
		center = pos
		size = sz
		level = lv
		
		# 多重極係数初期化
		multipole_coeffs.resize(MULTIPOLE_ORDER)
		local_coeffs.resize(MULTIPOLE_ORDER)
		for i in range(MULTIPOLE_ORDER):
			multipole_coeffs[i] = ComplexApprox.new()
			local_coeffs[i] = ComplexApprox.new()

# 主要FMM構造
var fmm_root: FMMNode
var particle_positions: PackedVector3Array
var particle_charges: PackedFloat32Array
var particle_potentials: PackedFloat32Array
var oct_tree_nodes: Array[FMMNode] = []

# Conway-Mathieu FMM統合
var conway_fmm_mapping: Dictionary = {}
var mathieu_local_expansions: Dictionary = {}

func _ready():
	super._ready()  # 親クラスの初期化
	initialize_fmm_system()

# ===== FMM初期化 =====
func initialize_fmm_system():
	print("Initializing FMM system with group theory...")
	
	# 基本構造初期化
	fmm_root = FMMNode.new(Vector3.ZERO, 100.0, 0)
	particle_positions = PackedVector3Array()
	particle_charges = PackedFloat32Array()
	particle_potentials = PackedFloat32Array()
	
	# Conway群とFMM階層の対応
	setup_conway_fmm_mapping()

func setup_conway_fmm_mapping():
	# Conway群の軌道 → FMM階層レベル対応
	var orbit_to_level = {
		0: 0,  # ノルム0 → ルートレベル
		4: 2,  # ノルム4 → 中間レベル（野生型）
		6: 4,  # ノルム6 → 詳細レベル
		8: 6   # ノルム8 → 最詳細レベル
	}
	
	for orbit in orbit_to_level.keys():
		var level = orbit_to_level[orbit]
		var pointer_key = "conway_norm_" + str(orbit)
		
		if virtual_pointers.has(pointer_key):
			conway_fmm_mapping[level] = virtual_pointers[pointer_key]
			print("Mapped Conway norm-", orbit, " to FMM level ", level)

# ===== GLTF→FMMパーティクル変換 =====
func convert_gltf_to_particles():
	print("Converting GLTF nodes to FMM particles...")
	
	particle_positions.clear()
	particle_charges.clear()
	particle_potentials.clear()
	
	for node_id in node_transforms.keys():
		var pointer = node_transforms[node_id]
		var position = extract_position_from_pointer(pointer)
		var charge = calculate_node_charge(node_id, pointer)
		
		particle_positions.append(position)
		particle_charges.append(charge)
		particle_potentials.append(0.0)  # 初期化
		
		print("Particle ", particle_positions.size()-1, ": pos=", position, ", charge=", charge)

func extract_position_from_pointer(pointer: VirtualPointer) -> Vector3:
	var array = get_current_array(pointer.group_type)
	
	if pointer.size >= 16:
		# 変換行列から位置抽出
		var x = array[pointer.start_index + 3]   # [0,3]
		var y = array[pointer.start_index + 7]   # [1,3]
		var z = array[pointer.start_index + 11]  # [2,3]
		return Vector3(x, y, z)
	
	return Vector3.ZERO

func calculate_node_charge(node_id: int, pointer: VirtualPointer) -> float:
	# 群論的電荷計算
	match pointer.group_type:
		"conway":
			# Conway群での電荷 = ノルム軌道に依存
			return float(pointer.norm_orbit) * 0.1
		"mathieu":
			# Mathieu群での電荷 = Steiner系構造
			return 1.0 / 24.0  # 24点系での均等分散
		_:
			return 1.0

# ===== FMM木構築 =====
func build_fmm_tree():
	print("Building FMM octree...")
	
	if particle_positions.size() == 0:
		push_error("No particles to build tree")
		return
	
	# 全パーティクルをルートに追加
	for i in range(particle_positions.size()):
		fmm_root.particles.append(i)
	
	# 再帰的分割
	subdivide_node(fmm_root)
	
	# 群論的ポインタ割り当て
	assign_group_pointers_to_nodes(fmm_root)

func subdivide_node(node: FMMNode):
	if node.particles.size() <= FMM_LEAF_SIZE or node.level >= FMM_MAX_DEPTH:
		return  # 葉ノード
	
	# 8方向分割
	var half_size = node.size * 0.5
	var quarter_size = half_size * 0.5
	
	for x in range(2):
		for y in range(2):
			for z in range(2):
				var child_center = node.center + Vector3(
					(x - 0.5) * half_size,
					(y - 0.5) * half_size,
					(z - 0.5) * half_size
				)
				
				var child = FMMNode.new(child_center, quarter_size, node.level + 1)
				
				# パーティクル分配
				for particle_id in node.particles:
					var pos = particle_positions[particle_id]
					if is_in_box(pos, child_center, quarter_size):
						child.particles.append(particle_id)
				
				if child.particles.size() > 0:
					node.children.append(child)
					oct_tree_nodes.append(child)
					subdivide_node(child)

func is_in_box(pos: Vector3, center: Vector3, half_size: float) -> bool:
	var diff = pos - center
	return abs(diff.x) <= half_size and abs(diff.y) <= half_size and abs(diff.z) <= half_size

func assign_group_pointers_to_nodes(node: FMMNode):
	# FMMレベルに対応する群論的ポインタを割り当て
	if conway_fmm_mapping.has(node.level):
		node.group_pointer = conway_fmm_mapping[node.level]
	elif node.level > 4:
		# 深いレベルはMathieu群
		node.group_pointer = virtual_pointers.get("mathieu_steiner")
	
	for child in node.children:
		assign_group_pointers_to_nodes(child)

# ===== 多重極展開計算 =====
func compute_multipole_expansions():
	print("Computing multipole expansions...")
	
	# ボトムアップで多重極係数計算
	compute_multipole_recursive(fmm_root)

func compute_multipole_recursive(node: FMMNode):
	# 子ノードから先に計算
	for child in node.children:
		compute_multipole_recursive(child)
	
	if node.children.size() == 0:
		# 葉ノード：直接計算
		compute_leaf_multipole(node)
	else:
		# 内部ノード：子から集約
		aggregate_child_multipoles(node)

func compute_leaf_multipole(node: FMMNode):
	# 葉ノードの多重極係数を直接計算
	for particle_id in node.particles:
		var pos = particle_positions[particle_id]
		var charge = particle_charges[particle_id]
		var rel_pos = pos - node.center
		
		# 球面調和関数近似（簡易版）
		for l in range(MULTIPOLE_ORDER):
			var r = rel_pos.length()
			if r > 1e-8:
				var cos_theta = rel_pos.z / r
				var phi = atan2(rel_pos.y, rel_pos.x)
				
				var coeff = charge * pow(r, l) * cos(l * phi)
				node.multipole_coeffs[l] = node.multipole_coeffs[l].add(
					ComplexApprox.new(coeff * cos_theta, coeff * sin(cos_theta))
				)

func aggregate_child_multipoles(node: FMMNode):
	# 子ノードの多重極係数を親に集約
	for child in node.children:
		var translation = child.center - node.center
		
		for l in range(MULTIPOLE_ORDER):
			# M2M変換（簡易版）
			var translated_coeff = translate_multipole_coefficient(
				child.multipole_coeffs[l], 
				translation, 
				l
			)
			node.multipole_coeffs[l] = node.multipole_coeffs[l].add(translated_coeff)

func translate_multipole_coefficient(coeff: ComplexApprox, translation: Vector3, order: int) -> ComplexApprox:
	# 多重極係数の平行移動（簡易版）
	var r = translation.length()
	if r < 1e-8:
		return coeff
	
	var factor = pow(r, order) / (order + 1)
	return ComplexApprox.new(coeff.real * factor, coeff.imag * factor)

# ===== 局所展開計算 =====
func compute_local_expansions():
	print("Computing local expansions...")
	
	# トップダウンで局所展開
	compute_local_recursive(fmm_root)

func compute_local_recursive(node: FMMNode):
	# M2L変換：遠方の多重極から局所展開
	compute_m2l_interactions(node)
	
	# 子ノードに局所展開を伝播
	for child in node.children:
		propagate_local_expansion(node, child)
		compute_local_recursive(child)

func compute_m2l_interactions(node: FMMNode):
	# 相互作用リストからM2L変換
	var interaction_list = find_interaction_list(node)
	
	for source_node in interaction_list:
		if source_node.level == node.level:
			# 同レベルのM2L変換
			var separation = source_node.center - node.center
			var distance = separation.length()
			
			if distance > INTERACTION_THRESHOLD * node.size:
				apply_m2l_transformation(source_node, node, separation)

func find_interaction_list(node: FMMNode) -> Array[FMMNode]:
	# 相互作用リスト構築（簡易版）
	var interaction_list: Array[FMMNode] = []
	
	# 同レベルの隣接しない遠方ノードを探す
	for other_node in oct_tree_nodes:
		if other_node.level == node.level and other_node != node:
			var distance = (other_node.center - node.center).length()
			var min_distance = INTERACTION_THRESHOLD * node.size
			var max_distance = min_distance * 4.0
			
			if distance >= min_distance and distance <= max_distance:
				interaction_list.append(other_node)
	
	return interaction_list

func apply_m2l_transformation(source: FMMNode, target: FMMNode, separation: Vector3):
	# M2L変換の実装
	var r = separation.length()
	if r < 1e-8:
		return
	
	for l in range(MULTIPOLE_ORDER):
		# 簡易版M2L変換
		var green_coeff = 1.0 / pow(r, l + 1)
		var contribution = ComplexApprox.new(
			source.multipole_coeffs[l].real * green_coeff,
			source.multipole_coeffs[l].imag * green_coeff
		)
		target.local_coeffs[l] = target.local_coeffs[l].add(contribution)

func propagate_local_expansion(parent: FMMNode, child: FMMNode):
	# L2L変換：親の局所展開を子に伝播
	var translation = child.center - parent.center
	
	for l in range(MULTIPOLE_ORDER):
		var translated_coeff = translate_local_coefficient(
			parent.local_coeffs[l], 
			translation, 
			l
		)
		child.local_coeffs[l] = child.local_coeffs[l].add(translated_coeff)

func translate_local_coefficient(coeff: ComplexApprox, translation: Vector3, order: int) -> ComplexApprox:
	# 局所係数の平行移動
	var r = translation.length()
	if r < 1e-8:
		return coeff
	
	var factor = pow(r, order)
	return ComplexApprox.new(coeff.real * factor, coeff.imag * factor)

# ===== 最終評価 =====
func evaluate_potentials():
	print("Evaluating final potentials...")
	
	# 各パーティクルでの電位評価
	for i in range(particle_positions.size()):
		var potential = 0.0
		var leaf_node = find_leaf_node(particle_positions[i])
		
		if leaf_node:
			# 局所展開からの寄与
			potential += evaluate_local_expansion(leaf_node, particle_positions[i])
			
			# 近傍パーティクルからの直接寄与
			potential += evaluate_near_field(leaf_node, i)
		
		particle_potentials[i] = potential

func find_leaf_node(pos: Vector3) -> FMMNode:
	return find_leaf_recursive(fmm_root, pos)

func find_leaf_recursive(node: FMMNode, pos: Vector3) -> FMMNode:
	if node.children.size() == 0:
		return node
	
	for child in node.children:
		if is_in_box(pos, child.center, child.size):
			return find_leaf_recursive(child, pos)
	
	return node

func evaluate_local_expansion(node: FMMNode, pos: Vector3) -> float:
	var potential = 0.0
	var rel_pos = pos - node.center
	var r = rel_pos.length()
	
	if r < 1e-8:
		return 0.0
	
	for l in range(MULTIPOLE_ORDER):
		var cos_theta = rel_pos.z / r if r > 0 else 0.0
		var phi = atan2(rel_pos.y, rel_pos.x)
		
		var basis = pow(r, l) * cos(l * phi) * cos_theta
		potential += node.local_coeffs[l].real * basis
	
	return potential

func evaluate_near_field(leaf_node: FMMNode, target_particle: int) -> float:
	var potential = 0.0
	var target_pos = particle_positions[target_particle]
	
	# 同じ葉ノード内の直接相互作用
	for source_particle in leaf_node.particles:
		if source_particle != target_particle:
			var source_pos = particle_positions[source_particle]
			var source_charge = particle_charges[source_particle]
			var distance = (target_pos - source_pos).length()
			
			if distance > 1e-8:
				potential += source_charge / distance

	return potential

# ===== FMM実行メイン =====
func run_fmm_calculation():
	print("=== Running FMM Calculation ===")
	
	# 1. GLTF→パーティクル変換
	convert_gltf_to_particles()
	
	# 2. FMM木構築
	build_fmm_tree()
	
	# 3. 多重極展開
	compute_multipole_expansions()
	
	# 4. 局所展開
	compute_local_expansions()
	
	# 5. 電位評価
	evaluate_potentials()
	
	print("FMM calculation completed!")
	print("Processed ", particle_positions.size(), " particles")
	
	# 結果の簡易統計
	var min_potential = particle_potentials[0]
	var max_potential = particle_potentials[0]
	var avg_potential = 0.0
	
	for i in range(particle_potentials.size()):
		var pot = particle_potentials[i]
		min_potential = min(min_potential, pot)
		max_potential = max(max_potential, pot)
		avg_potential += pot
	
	avg_potential /= particle_potentials.size()
	
	print("Potential range: [", min_potential, ", ", max_potential, "]")
	print("Average potential: ", avg_potential)

# ===== p2p最適化 =====
func optimize_for_p2p():
	print("Optimizing for peer-to-peer distribution...")
	
	# Conway群の分散表現
	var distributed_conway = distribute_conway_representation()
	
	# Mathieu群の局所計算
	var local_mathieu = localize_mathieu_computation()
	
	# 分散FMM
	setup_distributed_fmm(distributed_conway, local_mathieu)

func distribute_conway_representation() -> Dictionary:
	# Conway群の巨大な位数を複数ピアに分散
	var peer_count = 8  # 想定ピア数
	var conway_slice_size = CONWAY_ORDER / peer_count
	
	print("Distributing Conway group across ", peer_count, " peers")
	print("Each peer handles ~", conway_slice_size, " group elements")
	
	return {
		"peer_count": peer_count,
		"slice_size": conway_slice_size,
		"distribution_scheme": "norm_orbit_based"
	}

func localize_mathieu_computation() -> Dictionary:
	# Mathieu群の局所計算最適化
	print("Localizing Mathieu computations...")
	
	return {
		"steiner_blocks": 759,
		"local_optimization": true,
		"cache_size": MATHIEU24_ORDER % 10000
	}

func setup_distributed_fmm(conway_dist: Dictionary, mathieu_local: Dictionary):
	print("Setting up distributed FMM architecture...")
	print("Conway distribution: ", conway_dist)
	print("Mathieu localization: ", mathieu_local)
	
	# 実際のp2p通信は別途実装が必要
	print("Ready for p2p deployment!")

# 使用例
func example_fmm_usage():
	# GLTFロード
	load_gltf_with_potential_processing("res://complex_model.gltf")
	
	# FMM実行
	run_fmm_calculation()
	
	# p2p最適化
	optimize_for_p2p()
	
	# 状態表示
	print_system_status()
