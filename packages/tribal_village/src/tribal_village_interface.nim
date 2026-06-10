## Ultra-Fast Direct Buffer Interface
## Zero-copy numpy buffer communication - no conversions

import vmath
import common, environment, external_actions

var globalEnv: Environment = nil

proc tribal_village_create(): pointer {.exportc, dynlib.} =
  ## Create environment for direct buffer interface
  try:
    let config = defaultEnvironmentConfig()
    globalEnv = newEnvironment(config)
    initGlobalController(ExternalNN)
    return cast[pointer](globalEnv)
  except:
    return nil

proc tribal_village_reset_and_get_obs(
  env: pointer,
  obs_buffer: ptr UncheckedArray[uint8],    # [60, 21, 11, 11] direct
  rewards_buffer: ptr UncheckedArray[float32],
  terminals_buffer: ptr UncheckedArray[uint8],
  truncations_buffer: ptr UncheckedArray[uint8]
): int32 {.exportc, dynlib.} =
  ## Reset and write directly to buffers - no conversions
  if globalEnv == nil:
    return 0

  try:
    globalEnv.reset()
    if not globalEnv.observationsInitialized:
      globalEnv.rebuildObservations()

    # Direct memory copy of observations (zero conversion)
    let obs_size = MapAgents * ObservationLayers * ObservationWidth * ObservationHeight
    copyMem(obs_buffer, globalEnv.observations.addr, obs_size)

    # Clear rewards/terminals/truncations
    for i in 0..<MapAgents:
      rewards_buffer[i] = 0.0
      terminals_buffer[i] = 0
      truncations_buffer[i] = 0

    return 1
  except:
    return 0

proc tribal_village_step_with_pointers(
  env: pointer,
  actions_buffer: ptr UncheckedArray[uint8],    # [MapAgents] direct read
  obs_buffer: ptr UncheckedArray[uint8],        # [60, 21, 11, 11] direct write
  rewards_buffer: ptr UncheckedArray[float32],
  terminals_buffer: ptr UncheckedArray[uint8],
  truncations_buffer: ptr UncheckedArray[uint8]
): int32 {.exportc, dynlib.} =
  ## Ultra-fast step with direct buffer access
  if globalEnv == nil:
    return 0

  try:
    # Read actions directly from buffer (no conversion)
    var actions: array[MapAgents, uint8]
    for i in 0..<MapAgents:
      actions[i] = actions_buffer[i]

    # Step environment
    globalEnv.step(unsafeAddr actions)

    # Direct memory copy of observations (zero conversion overhead)
    let obs_size = MapAgents * ObservationLayers * ObservationWidth * ObservationHeight
    copyMem(obs_buffer, globalEnv.observations.addr, obs_size)

    # Direct buffer writes (no dict conversion)
    for i in 0..<MapAgents:
      let agent = (if i < globalEnv.agents.len: globalEnv.agents[i] else: nil)
      let reward = if agent.isNil: 0.0'f32 else: agent.reward
      rewards_buffer[i] = reward
      if not agent.isNil:
        agent.reward = 0.0'f32
      terminals_buffer[i] = if globalEnv.terminated[i] > 0.0: 1 else: 0
      truncations_buffer[i] = if globalEnv.truncated[i] > 0.0: 1 else: 0

    return 1
  except:
    return 0

proc tribal_village_get_action_stats(
  env: pointer,
  stats_buffer: ptr UncheckedArray[int32]  # [MapAgents, 28]
): int32 {.exportc, dynlib.} =
  ## Copy cumulative action stats to a flat buffer.
  ##
  ## Columns are mirrored in v3_experiments.tribal_behavior.SIMULATOR_STAT_COLUMNS.
  if globalEnv == nil or stats_buffer.isNil:
    return 0

  try:
    for i in 0..<MapAgents:
      let offset = i * 32
      if i < globalEnv.stats.len:
        let stats = globalEnv.stats[i]
        stats_buffer[offset + 0] = stats.actionInvalid.int32
        stats_buffer[offset + 1] = stats.actionNoop.int32
        stats_buffer[offset + 2] = stats.actionMove.int32
        stats_buffer[offset + 3] = stats.actionAttack.int32
        stats_buffer[offset + 4] = stats.actionUse.int32
        stats_buffer[offset + 5] = stats.actionSwap.int32
        stats_buffer[offset + 6] = stats.actionPut.int32
        stats_buffer[offset + 7] = stats.actionPlant.int32
        stats_buffer[offset + 8] = stats.resourceWater.int32
        stats_buffer[offset + 9] = stats.resourceWheat.int32
        stats_buffer[offset + 10] = stats.resourceWood.int32
        stats_buffer[offset + 11] = stats.resourceOre.int32
        stats_buffer[offset + 12] = stats.craftBattery.int32
        stats_buffer[offset + 13] = stats.craftSpear.int32
        stats_buffer[offset + 14] = stats.craftLantern.int32
        stats_buffer[offset + 15] = stats.craftArmor.int32
        stats_buffer[offset + 16] = stats.craftBread.int32
        stats_buffer[offset + 17] = stats.depositHeart.int32
        stats_buffer[offset + 18] = stats.putArmor.int32
        stats_buffer[offset + 19] = stats.putBread.int32
        stats_buffer[offset + 20] = stats.tumorKill.int32
        stats_buffer[offset + 21] = stats.spawnerKill.int32
        stats_buffer[offset + 22] = stats.agentKill.int32
        stats_buffer[offset + 23] = stats.death.int32
        stats_buffer[offset + 24] = stats.respawn.int32
        stats_buffer[offset + 25] = stats.lanternPlant.int32
        stats_buffer[offset + 26] = stats.putOre.int32
        stats_buffer[offset + 27] = stats.putBattery.int32
        stats_buffer[offset + 28] = stats.putOreToCrafter.int32
        stats_buffer[offset + 29] = stats.putBatteryToDepositor.int32
        stats_buffer[offset + 30] = stats.receiveOreFromSupplier.int32
        stats_buffer[offset + 31] = stats.receiveBatteryFromCrafter.int32
      else:
        for col in 0..<32:
          stats_buffer[offset + col] = 0
    return 1
  except:
    return 0

proc tribal_village_get_inventory_snapshot(
  env: pointer,
  inventory_buffer: ptr UncheckedArray[int32]  # [MapAgents, 9]
): int32 {.exportc, dynlib.} =
  ## Copy current per-agent inventory counts.
  if globalEnv == nil or inventory_buffer.isNil:
    return 0

  try:
    for i in 0..<MapAgents:
      let offset = i * 9
      if i < globalEnv.agents.len:
        let agent = globalEnv.agents[i]
        inventory_buffer[offset + 0] = agent.inventoryOre.int32
        inventory_buffer[offset + 1] = agent.inventoryBattery.int32
        inventory_buffer[offset + 2] = agent.inventoryWater.int32
        inventory_buffer[offset + 3] = agent.inventoryWheat.int32
        inventory_buffer[offset + 4] = agent.inventoryWood.int32
        inventory_buffer[offset + 5] = agent.inventorySpear.int32
        inventory_buffer[offset + 6] = agent.inventoryLantern.int32
        inventory_buffer[offset + 7] = agent.inventoryArmor.int32
        inventory_buffer[offset + 8] = agent.inventoryBread.int32
      else:
        for col in 0..<9:
          inventory_buffer[offset + col] = 0
    return 1
  except:
    return 0

proc tribal_village_get_world_stats(
  env: pointer,
  world_buffer: ptr UncheckedArray[int32]  # [12]
): int32 {.exportc, dynlib.} =
  ## Copy coarse world-state counters.
  if globalEnv == nil or world_buffer.isNil:
    return 0

  try:
    var liveAgents = 0
    var deadAgents = 0
    var assemblers = 0
    var assemblerHearts = 0
    var mines = 0
    var converters = 0
    var spawners = 0
    var tumors = 0
    var plantedLanterns = 0
    var productionBuildings = 0

    for i in 0..<MapAgents:
      if globalEnv.terminated[i] > 0.0:
        inc deadAgents
      else:
        inc liveAgents

    for thing in globalEnv.things:
      case thing.kind
      of assembler:
        inc assemblers
        assemblerHearts += thing.hearts
      of Mine:
        inc mines
      of Converter:
        inc converters
      of Spawner:
        inc spawners
      of Tumor:
        inc tumors
      of PlantedLantern:
        inc plantedLanterns
      of Forge, Armory, ClayOven, WeavingLoom:
        inc productionBuildings
      else:
        discard

    world_buffer[0] = globalEnv.currentStep.int32
    world_buffer[1] = liveAgents.int32
    world_buffer[2] = deadAgents.int32
    world_buffer[3] = assemblers.int32
    world_buffer[4] = assemblerHearts.int32
    world_buffer[5] = mines.int32
    world_buffer[6] = converters.int32
    world_buffer[7] = spawners.int32
    world_buffer[8] = tumors.int32
    world_buffer[9] = plantedLanterns.int32
    world_buffer[10] = productionBuildings.int32
    world_buffer[11] = globalEnv.things.len.int32
    return 1
  except:
    return 0

proc nearestThingSnapshot(env: Environment, pos: IVec2, kind: ThingKind): tuple[x: int32, y: int32, dist: int32] =
  result = (x: -1'i32, y: -1'i32, dist: -1'i32)
  var bestDist = int.high
  for thing in env.things:
    if thing.kind != kind:
      continue
    let distance = manhattanDistance(pos, thing.pos)
    if distance < bestDist:
      bestDist = distance
      result = (x: thing.pos.x.int32, y: thing.pos.y.int32, dist: distance.int32)

proc tribal_village_get_navigation_snapshot(
  env: pointer,
  navigation_buffer: ptr UncheckedArray[int32]  # [MapAgents, 13]
): int32 {.exportc, dynlib.} =
  ## Copy per-agent navigation state for reward breadcrumbs and debugging.
  if globalEnv == nil or navigation_buffer.isNil:
    return 0

  try:
    for i in 0..<MapAgents:
      let offset = i * 13
      if i < globalEnv.agents.len:
        let agent = globalEnv.agents[i]
        let nearestConverter = nearestThingSnapshot(globalEnv, agent.pos, Converter)
        let nearestMine = nearestThingSnapshot(globalEnv, agent.pos, Mine)
        var homeDist = -1'i32
        if agent.homeassembler.x >= 0 and agent.homeassembler.y >= 0:
          homeDist = manhattanDistance(agent.pos, agent.homeassembler).int32
        navigation_buffer[offset + 0] = agent.pos.x.int32
        navigation_buffer[offset + 1] = agent.pos.y.int32
        navigation_buffer[offset + 2] = agent.homeassembler.x.int32
        navigation_buffer[offset + 3] = agent.homeassembler.y.int32
        navigation_buffer[offset + 4] = nearestConverter.x
        navigation_buffer[offset + 5] = nearestConverter.y
        navigation_buffer[offset + 6] = nearestMine.x
        navigation_buffer[offset + 7] = nearestMine.y
        navigation_buffer[offset + 8] = homeDist
        navigation_buffer[offset + 9] = nearestConverter.dist
        navigation_buffer[offset + 10] = nearestMine.dist
        navigation_buffer[offset + 11] = agent.inventoryOre.int32
        navigation_buffer[offset + 12] = agent.inventoryBattery.int32
      else:
        for col in 0..<13:
          navigation_buffer[offset + col] = -1
    return 1
  except:
    return 0

proc tribal_village_get_action_mask(
  env: pointer,
  mask_buffer: ptr UncheckedArray[uint8]  # [MapAgents, ActionVerbCount * ActionArgumentCount]
): int32 {.exportc, dynlib.} =
  ## Copy a per-agent mask of actions that can currently succeed.
  if globalEnv == nil or mask_buffer.isNil:
    return 0

  try:
    let actionCount = ActionVerbCount * ActionArgumentCount
    for i in 0..<MapAgents:
      let offset = i * actionCount
      for action in 0..<actionCount:
        mask_buffer[offset + action] =
          if globalEnv.isActionCurrentlyValid(i, action.uint8): 1'u8 else: 0'u8
    return 1
  except:
    return 0

proc tribal_village_get_num_agents(): int32 {.exportc, dynlib.} =
  return MapAgents.int32

proc tribal_village_get_obs_layers(): int32 {.exportc, dynlib.} =
  return ObservationLayers.int32

proc tribal_village_get_obs_width(): int32 {.exportc, dynlib.} =
  return ObservationWidth.int32


proc tribal_village_get_map_width(): int32 {.exportc, dynlib.} =
  return MapWidth.int32

proc tribal_village_get_map_height(): int32 {.exportc, dynlib.} =
  return MapHeight.int32

# Render full map as HxWx3 RGB (uint8)
proc toByte(value: float32): uint8 =
  var iv = int(value * 255.0)
  if iv < 0:
    iv = 0
  elif iv > 255:
    iv = 255
  result = uint8(iv)

proc tribal_village_render_rgb(
  env: pointer,
  out_buffer: ptr UncheckedArray[uint8],
  out_w: int32,
  out_h: int32
): int32 {.exportc, dynlib.} =
  if globalEnv == nil or out_buffer.isNil:
    return 0

  let width = int(out_w)
  let height = int(out_h)
  if width <= 0 or height <= 0:
    return 0
  if width mod MapWidth != 0 or height mod MapHeight != 0:
    return 0

  let scaleX = width div MapWidth
  let scaleY = height div MapHeight
  let stride = width * 3

  try:
    for y in 0 ..< MapHeight:
      for sy in 0 ..< scaleY:
        let rowBase = (y * scaleY + sy) * stride
        for x in 0 ..< MapWidth:
          var rByte = toByte(globalEnv.tileColors[x][y].r)
          var gByte = toByte(globalEnv.tileColors[x][y].g)
          var bByte = toByte(globalEnv.tileColors[x][y].b)

          let thing = globalEnv.grid[x][y]
          if thing != nil:
            case thing.kind
            of Agent:
              rByte = 255'u8
              gByte = 255'u8
              bByte = 0'u8
            of Tumor:
              rByte = 160'u8
              gByte = 32'u8
              bByte = 240'u8
            of Wall:
              rByte = 96'u8
              gByte = 96'u8
              bByte = 96'u8
            of Mine:
              rByte = 184'u8
              gByte = 134'u8
              bByte = 11'u8
            of Converter:
              rByte = 0'u8
              gByte = 200'u8
              bByte = 200'u8
            of assembler:
              rByte = 220'u8
              gByte = 0'u8
              bByte = 220'u8
            of Spawner:
              rByte = 255'u8
              gByte = 170'u8
              bByte = 0'u8
            of Armory:
              rByte = 255'u8
              gByte = 120'u8
              bByte = 40'u8
            of Forge:
              rByte = 255'u8
              gByte = 80'u8
              bByte = 0'u8
            of ClayOven:
              rByte = 255'u8
              gByte = 180'u8
              bByte = 120'u8
            of WeavingLoom:
              rByte = 0'u8
              gByte = 180'u8
              bByte = 255'u8
            of PlantedLantern:
              rByte = 255'u8
              gByte = 240'u8
              bByte = 128'u8
            else:
              discard

          let xBase = rowBase + x * scaleX * 3
          for sx in 0 ..< scaleX:
            let idx = xBase + sx * 3
            out_buffer[idx] = rByte
            out_buffer[idx + 1] = gByte
            out_buffer[idx + 2] = bByte
    return 1
  except:
    return 0
proc tribal_village_get_obs_height(): int32 {.exportc, dynlib.} =
  return ObservationHeight.int32

proc tribal_village_destroy(env: pointer) {.exportc, dynlib.} =
  ## Clean up environment
  globalEnv = nil

# --- Rendering interface (ANSI) ---
proc tribal_village_render_ansi(
  env: pointer,
  out_buffer: ptr UncheckedArray[char],
  buf_len: int32
): int32 {.exportc, dynlib.} =
  ## Write an ANSI string render into out_buffer (null-terminated).
  ## Returns number of bytes written (excluding terminator). 0 on error.
  if globalEnv == nil or out_buffer.isNil or buf_len <= 1:
    return 0

  try:
    let s = render(globalEnv)  # environment.render*(env: Environment): string
    let n = min(s.len, max(0, buf_len - 1).int)
    if n > 0:
      copyMem(out_buffer, cast[pointer](s.cstring), n)
    out_buffer[n] = '\0'  # null-terminate
    return n.int32
  except:
    return 0
