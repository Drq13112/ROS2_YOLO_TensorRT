#include "perception_obstacles/perception_obstacles_node.hpp"

/** ----------------------------------------------------------------------------------------------------------------<
 * @brief Constructor de la clase
 *
 */
PerceptionObstacles::PerceptionObstacles() : Node("perception_obstacles")
{
  printf("INITIALIZING VARIABLES ... \n");

  // # Increase the maximum receive buffer size for network packets
  // sudo nano /etc/sysctl.conf
  // net.core.rmem_max=2147483647  # 2 GiB, default is 208 KiB
  // net.core.rmem_default=2147483647

  // ---------- Quality of Service ---------- //qos
  // We are selecting the default configuration for sensors (key_word: rmw_qos_profile_sensor_data) with the only
  // modification of changing depth from 5 to 1 and "reliable" because we are loosing less msgs.

  rclcpp::QoS qos_sensors(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
  // qos_sensors.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);  // CARE! Publisher MUST be reliable too
  qos_sensors.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
  qos_sensors.keep_last(1);  // depth = 1

  rclcpp::QoS qos_odom(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
  qos_odom.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
  qos_odom.keep_last(1);  // depth = 1

  rclcpp::QoS qos_publish_debug(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
  qos_publish_debug.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos_publish_debug.keep_last(1);  // depth = 1

  // ---------- Subscritors and Publishers ---------- //

  printf("Initializing subscriptors and publishers ... \n");

  int topics_found = 0;
  // Create Ruby subscription
  std::string topic_RubyPlus = this->declare_parameter("topic_RubyPlus", "rubyplus_points");
  topics_found = PerceptionObstacles::check_QoS_publisher_s(topic_RubyPlus);
  if (topics_found == 0)
  {
    printf("!!!!!!!!!! Check that Ruby Plus is publishing !!!!!!!!!! \n");
  }
  this->RubyPlus_subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      topic_RubyPlus, qos_sensors, std::bind(&PerceptionObstacles::RubyPlusCallback, this, std::placeholders::_1));

  // Create localization subscription
  std::string topic_localization = this->declare_parameter("topic_localization", "zoe/localization/global");
  topics_found = PerceptionObstacles::check_QoS_publisher_s(topic_localization);
  if (topics_found == 0)
  {
    printf("!!!!!!!!!! Check that Localization is being published !!!!!!!!!! \n");
  }
  this->localization_subscription = this->create_subscription<nav_msgs::msg::Odometry>(
      topic_localization, qos_odom, std::bind(&PerceptionObstacles::LocalizationCallback, this, std::placeholders::_1));

  // Create Helios subscription
  if (consider_PC_HeliosRight_)
  {
    std::string topic_HeliosRight = this->declare_parameter("topic_HeliosRight_points", "helios_right_points");
    PerceptionObstacles::check_QoS_publisher_s(topic_HeliosRight);
    this->HeliosRight_subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        topic_HeliosRight, qos_sensors,
        std::bind(&PerceptionObstacles::HeliosRightCallback, this, std::placeholders::_1));
  }

  if (consider_PC_HeliosLeft_)
  {
    std::string topic_HeliosLeft = this->declare_parameter("topic_HeliosLeft_points", "helios_left_points");
    PerceptionObstacles::check_QoS_publisher_s(topic_HeliosLeft);
    this->HeliosLeft_subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        topic_HeliosLeft, qos_sensors,
        std::bind(&PerceptionObstacles::HeliosLeftCallback, this, std::placeholders::_1));
  }

  // Create publishers
  std::string topic_classified_RB = this->declare_parameter("topic_publish_classified_RubyPlus", "classified_RubyPlus");
  pc_RB_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_classified_RB, qos_publish_debug);

  if (consider_PC_HeliosRight_)
  {
    std::string topic_classified_Hr =
        this->declare_parameter("topic_publish_classified_HeliosRight", "classified_HeliosRight");
    pc_Hr_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_classified_Hr, qos_publish_debug);
  }
  if (consider_PC_HeliosLeft_)
  {
    std::string topic_classified_Hl =
        this->declare_parameter("topic_publish_classified_HeliosLeft", "classified_HeliosLeft");
    pc_Hl_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_classified_Hl, qos_publish_debug);
  }

  std::string topic_obs_pO = this->declare_parameter("topic_publish_obs_pO", "obs_pO");
  obs_OG_pO_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(topic_obs_pO, qos_publish_debug);

  // std::string topic_obs_mO = this->declare_parameter("obs_mO_topic", "obs_mO");
  // obs_OG_mO_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(topic_obs_mO, qos_publish_debug);

  // std::string topic_obs_mF = this->declare_parameter("obs_mF_topic", "obs_mF");
  // obs_OG_mF_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(topic_obs_mF, qos_publish_debug);

  tf_odom_world_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  printf("... initialization subscriptors and publishers done \n\n");

  // ---------- Load Parameters and Initialize ---------- //
  PerceptionObstacles::GetTransforms();
  write_files_time(&TIME_measurements_, true);
  EGO_VEH::write_files_localization(&debug_received_info_coche_, true);

  // ---------- Localization ---------- //
  checkCudaErrors(cudaMalloc((void**)&d_info_coche_, sizeof(EGO_VEH::INFO_ego)));  // Reserve

  // ---------- CART GRID (basic) ---------- //

  checkCudaErrors(cudaMallocHost((void**)&h_grid_cart_data_, sizeof(GRID_TYPES::CART_Data)));  // Reserve
  checkCudaErrors(cudaMalloc((void**)&d_grid_cart_data_, sizeof(GRID_TYPES::CART_Data)));      // Reserve

  grid_inicializar_rejilla_cartesiana(h_grid_cart_data_);

  checkCudaErrors(cudaMemcpy(d_grid_cart_data_, h_grid_cart_data_, sizeof(GRID_TYPES::CART_Data),
                             cudaMemcpyHostToDevice));  // cudaMemcpy() blocks the host until the memory transfer and
                                                        // all prior operations in the same stream are completed.

  cudaDeviceSynchronize();  // Not needed because cudaMemcpy alredy blocks, but just in case

  // ---------- DOG + obs OG ---------- //
  checkCudaErrors(cudaMallocHost((void**)&h_grid_obs_, sizeof(GRID_TYPES::OG)));  // Reserve
  checkCudaErrors(cudaMalloc((void**)&d_grid_obs_, sizeof(GRID_TYPES::OG)));      // Reserve

  checkCudaErrors(cudaMallocHost((void**)&h_grid_, sizeof(GRID_TYPES::DOG)));  // Reserve
  checkCudaErrors(cudaMalloc((void**)&d_grid_, sizeof(GRID_TYPES::DOG)));      // Reserve

  checkCudaErrors(cudaMallocHost((void**)&h_particles_, sizeof(PARTICLE_TYPES::PART_DOG)));             // Reserve
  checkCudaErrors(cudaMalloc((void**)&d_particles_, sizeof(PARTICLE_TYPES::PART_DOG)));                 // Reserve
  checkCudaErrors(cudaMalloc((void**)&d_particles_sorted_, sizeof(PARTICLE_TYPES::PART_DOG)));          // Reserve
  checkCudaErrors(cudaMalloc((void**)&d_particles_for_resampling_, sizeof(PARTICLE_TYPES::PART_DOG)));  // Reserve

  memset(&h_particles_->valida, false, PARTICLE_TYPES::NP_TOT * sizeof(bool));
  checkCudaErrors(cudaMemcpy(d_particles_, h_particles_, sizeof(PARTICLE_TYPES::PART_DOG), cudaMemcpyHostToDevice));
  memset(&h_grid_->masa_pred_oc_factor, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(double));
  memset(&h_grid_->masa_pred_libre_factor, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(double));
  memset(&h_grid_->masa_act_oc_factor, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(double));
  memset(&h_grid_->masa_act_libre_factor, 0.0, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(double));
  checkCudaErrors(cudaMemcpy(d_grid_, h_grid_, sizeof(GRID_TYPES::DOG), cudaMemcpyHostToDevice));

  initial_rng_seed = time(NULL);
  checkCudaErrors(cudaMalloc((float**)&d_random_pred, 2 * PARTICLE_TYPES::NP_ACT * sizeof(float)));
  checkCudaErrors(cudaMalloc((float**)&d_random_particle_selection, PARTICLE_TYPES::NP_ACT * sizeof(float)));
  checkCudaErrors(cudaMalloc((float**)&d_random_cell_selection, PARTICLE_TYPES::NP_NEW * sizeof(float)));
  checkCudaErrors(cudaMalloc((float**)&d_random_asociacion, PARTICLE_TYPES::NP_NEW * sizeof(float)));
  checkCudaErrors(cudaMalloc((float**)&d_random_vel_uniforme, 2 * PARTICLE_TYPES::NP_NEW * sizeof(float)));

  checkCudaErrors(cudaMallocHost((void**)&h_config_DOG_, sizeof(DYN_CLASS_OG::config)));
  checkCudaErrors(cudaMalloc((void**)&d_config_DOG_, sizeof(DYN_CLASS_OG::config)));
  DYN_CLASS_OG::leer_fichero_configuracion_DOG(h_config_DOG_);
  checkCudaErrors(cudaMemcpy(d_config_DOG_, h_config_DOG_, sizeof(DYN_CLASS_OG::config), cudaMemcpyHostToDevice));

  // ---------- RubyPlus ---------- //
  printf("\n");
  this->RubyPlusLoadParameters();
  AUTOPIA_RubyPlus::initialize_metachannels_data(&RB_data_);

  AUTOPIA_RubyPlus::initialize_pointcloud(&RB_pc_, &RB_data_, 1, RubyPlus_transform_->transform.translation.x,
                                          RubyPlus_transform_->transform.translation.y,
                                          RubyPlus_transform_->transform.translation.z, AUTOPIA_RubyPlus::N_POINTS,
                                          RB_data_.n_points_metachannel, RubyPlus_transform_->transform);
  memcpy(&RB_pc_modificable_callback_, &RB_pc_, sizeof(RB_pc_));

  checkCudaErrors(cudaMalloc((void**)&d_RB_pc_, sizeof(AUTOPIA_RubyPlus::PointCloud)));  // Reserve
  checkCudaErrors(cudaMemcpy(d_RB_pc_, &RB_pc_, sizeof(AUTOPIA_RubyPlus::PointCloud), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void**)&d_RB_PolarOG_, sizeof(GRID_TYPES::POLAR_OG)));  // Reserve

  printf("Ruby Plus initialized\n\n");

  // ---------- Helios ---------- //
  printf("\n");
  this->HeliosLoadParameters();
  AUTOPIA_Helios::initialize_metachannels_data(&Helios_data_);

  // Helios Right
  AUTOPIA_Helios::initialize_pointcloud(&Hr_pc_, &Helios_data_, 2, HeliosRight_transform_->transform.translation.x,
                                        HeliosRight_transform_->transform.translation.y,
                                        HeliosRight_transform_->transform.translation.z, AUTOPIA_Helios::N_POINTS,
                                        Helios_data_.n_points_metachannel, HeliosRight_transform_->transform);

  memcpy(&Hr_pc_modificable_callback_, &Hr_pc_, sizeof(Hr_pc_));

  checkCudaErrors(cudaMalloc((void**)&d_Hr_pc_, sizeof(AUTOPIA_Helios::PointCloud)));  // Reserve

  checkCudaErrors(cudaMemcpy(d_Hr_pc_, &Hr_pc_, sizeof(AUTOPIA_Helios::PointCloud), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void**)&d_Hr_PolarOG_, sizeof(GRID_TYPES::POLAR_OG_small)));  // Reserve
  printf("Helios Right initialized\n\n");

  // Helios left
  AUTOPIA_Helios::initialize_pointcloud(&Hl_pc_, &Helios_data_, 3, HeliosLeft_transform_->transform.translation.x,
                                        HeliosLeft_transform_->transform.translation.y,
                                        HeliosLeft_transform_->transform.translation.z, AUTOPIA_Helios::N_POINTS,
                                        Helios_data_.n_points_metachannel, HeliosLeft_transform_->transform);

  memcpy(&Hl_pc_modificable_callback_, &Hl_pc_, sizeof(Hl_pc_));

  checkCudaErrors(cudaMalloc((void**)&d_Hl_pc_, sizeof(AUTOPIA_Helios::PointCloud)));  // Reserve
  checkCudaErrors(cudaMemcpy(d_Hl_pc_, &Hl_pc_, sizeof(AUTOPIA_Helios::PointCloud), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void**)&d_Hl_PolarOG_, sizeof(GRID_TYPES::POLAR_OG_small)));  // Reserve
  printf("Helios Left initialized\n\n");

  // ---------- Offline Road map ---------- //
  printf("\n\nInitializing offline road map ... \n");

  // Config
  checkCudaErrors(cudaMallocHost((void**)&h_config_map_, sizeof(OFF_ROAD_MAP::config)));
  checkCudaErrors(cudaMalloc((void**)&d_config_map_, sizeof(OFF_ROAD_MAP::config)));
  OFF_ROAD_MAP::leer_fichero_configuracion_mapa_ruta(h_config_map_);
  checkCudaErrors(cudaMemcpy(d_config_map_, h_config_map_, sizeof(OFF_ROAD_MAP::config), cudaMemcpyHostToDevice));

  // Road map offline
  checkCudaErrors(cudaMallocHost((void**)&h_complete_road_map_, h_config_map_->NC_XY * sizeof(char)));
  checkCudaErrors(cudaMalloc((void**)&d_complete_road_map_, h_config_map_->NC_XY * sizeof(char)));

  OFF_ROAD_MAP::carga_mapa(h_complete_road_map_, h_config_map_);
  checkCudaErrors(cudaMemcpy(d_complete_road_map_, h_complete_road_map_, h_config_map_->NC_XY * sizeof(char),
                             cudaMemcpyHostToDevice));

  // Local grid
  checkCudaErrors(cudaMallocHost((void**)&h_grid_road_, sizeof(GRID_TYPES::GRID_local_road)));
  checkCudaErrors(cudaMalloc((void**)&d_grid_road_, sizeof(GRID_TYPES::GRID_local_road)));
  std::memset(&h_grid_road_->es_carretera, false, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(bool));
  checkCudaErrors(cudaMemset(&d_grid_road_->es_carretera, false, GRID_TYPES::NC_Y * GRID_TYPES::NC_X * sizeof(bool)));

  // OFF_ROAD_MAP::initialize_offline_road_map(h_config_map_, d_config_map_, h_complete_road_map_, d_complete_road_map_,
  //                                           h_grid_road_, d_grid_road_);
  // cudaDeviceSynchronize();

  printf("... offline road map initialized \n\n");

  // ---------- SYNCHRONIZE INITIALIZATION ---------- //
  cudaDeviceSynchronize();  // Just in case...

  printf("... INITIALIZATION DONE \n\n\n");

  // ---------- CUDA STUFF ---------- //

  for (int i = 0; i < num_streams_; i++)
  {
    cudaStreamCreate(&streams_[i]);
  }

  printf("CUDA STUFF: \n");
  cudaMemGetInfo(&freeMem, &totalMem);
  std::cout << " - Free memory: " << freeMem << " bytes, Total memory: " << totalMem << " bytes" << std::endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << " - Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << " - Max Shared Memory per Block: " << prop.sharedMemPerBlock << std::endl;
  std::cout << " - Max Registers per Block: " << prop.regsPerBlock << std::endl;

  printf("\n\n\n");

  // ---------- LAUNCH MAIN THREAD ---------- //

  main_thread = std::make_unique<std::thread>(&PerceptionObstacles::Run, this);
}

/** ----------------------------------------------------------------------------------------------------------------<
 * @brief Destructor de la clase
 *
 */
PerceptionObstacles::~PerceptionObstacles()
{
  // Localization
  cudaFree(d_info_coche_);

  // RubyPlus
  cudaFree(d_RB_pc_);
  cudaFree(d_RB_PolarOG_);
  // cudaFree(d_RB_CartOG_);

  cudaFree(d_Hr_pc_);
  cudaFree(d_Hr_PolarOG_);
  // cudaFree(d_Hr_CartOG_);

  cudaFree(d_Hl_pc_);
  cudaFree(d_Hl_PolarOG_);
  // cudaFree(d_Hl_CartOG_);

  // Grid Cart
  cudaFreeHost(h_grid_cart_data_);
  cudaFree(d_grid_cart_data_);

  // Obs grid
  cudaFreeHost(h_grid_obs_);
  cudaFree(d_grid_obs_);

  // DOG
  cudaFreeHost(h_particles_);
  cudaFree(d_particles_);
  cudaFree(d_particles_sorted_);
  cudaFree(d_particles_for_resampling_);
  cudaFreeHost(h_grid_);
  cudaFree(d_grid_);

  cudaFree(d_random_pred);
  cudaFree(d_random_particle_selection);
  cudaFree(d_random_cell_selection);
  cudaFree(d_random_asociacion);
  cudaFree(d_random_vel_uniforme);

  // End thread
  main_thread->join();
}

/** ----------------------------------------------------------------------------------------------------------------<
 * @brief Load parameters RubyPlus
 *
 */
void PerceptionObstacles::RubyPlusLoadParameters()
{
  printf("Loading RubyPlus parameters...\n");

  RB_data_.rows = this->declare_parameter("ruby_rows", 1800);

  RB_data_.cols = this->declare_parameter("ruby_cols", 128);

  RB_data_.groups = this->declare_parameter("ruby_groups", 8);

  RB_data_.shift_metachannel = this->declare_parameter("ruby_shift", 60);

  RB_data_.n_subchannels = this->declare_parameter("ruby_subchannels_count", 14400);

  RB_data_.n_points_subchannel = this->declare_parameter("ruby_subchannel_points", 16);

  RB_data_.n_metachannels = this->declare_parameter("ruby_metachannels_count", 7200);

  RB_data_.n_points_metachannel = this->declare_parameter("ruby_metachannel_points", 32);

  printf("... RubyPlus parameters loaded\n");
}

void PerceptionObstacles::HeliosLoadParameters()
{
  printf("Loading Helios parameters...\n");

  Helios_data_.rows = this->declare_parameter("helios_rows", 1800);

  Helios_data_.cols = this->declare_parameter("helios_cols", 32);

  Helios_data_.groups = this->declare_parameter("helios_groups", 2);

  Helios_data_.n_metachannels = this->declare_parameter("helios_metachannels_count", 3600);

  Helios_data_.n_points_metachannel = this->declare_parameter("helios_metachannel_points", 16);

  printf("... Helios parameters loaded\n");
}

/** ----------------------------------------------------------------------------------------------------------------
 * @brief Hilo principal que se suscribe y ejecuta los callbacks
 * 1. Espera a que haya disponible una nube de puntos de Ruby Plus, cuando la hay la almacena.
 */
void PerceptionObstacles::Run()
{
  ChronoTimer timer_iteration;
  int iter_inicial = 1;
  int iteration = iter_inicial;

  printf("\n\n------------------------------ Empieza bucle ------------------------------\n");

  bool enumeration = true;

  // +++++++++++++++++++++++++++++++++++++++++++++ Wait sources +++++++++++++++++++++++++++++++++++++++++++++ //

  std::unique_lock<std::mutex> aux_lock(mutex_);

  // Lets wait for localization
  printf("Waiting for localization...\n");

  init_localization_condition_.wait(aux_lock, [this] { return flag_avoid_spurious_localization_callback_ == true; });
  memcpy(&h_info_coche_, &info_coche_modificable_callback_, sizeof(info_coche_modificable_callback_));
  TIME_measurements_.n_odom_callbacks = 0;
  aux_lock.unlock();

  printf("... confirmed localization active\n");

  // Confirm there is ruby plus data
  printf("Waiting for RubyPlus... \n");

  aux_lock.lock();
  RubyPlus_condition_.wait(aux_lock, [this] { return flag_avoid_spurious_RubyPlus_callback_ == true; });
  flag_avoid_spurious_RubyPlus_callback_ = false;
  TIME_measurements_.n_RubyPlus_callbacks = 0;
  aux_lock.unlock();

  printf("... confirmed RubyPlus active\n");

  // Lets sleep for 1 sec so confirm that we have some data from our asynchronous sources (max delta 0.1sec, so... )
  sleep(1);

  // +++++++++++++++++++++++++++++++++++++++++++++ Loop +++++++++++++++++++++++++++++++++++++++++++++ //
  while (rclcpp::ok())
  {
    // ++++++++++++++++++++++++++++++++++++++++++ Synchronization ++++++++++++++++++++++++++++++++++++++++++ //

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! locking !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! //

    printf(" ... esperando nuevo ruby ... \n");
    // Wait for RubyPlus (currently the mutex is locked, the wait unlocks it (lets other threads do stuff) and waits to
    // be notify to lock it again (and this thread takes control again))
    std::unique_lock<std::mutex> lockRubyPlus(mutex_);
    RubyPlus_condition_.wait(lockRubyPlus, [this] { return flag_avoid_spurious_RubyPlus_callback_ == true; });

    if (enumeration)
    {
      TIME_measurements_.time_total_iteration.Reset();
      printf("\n --------------- Iter %d (Ruby cb %d) --------------- \n", iteration,
             TIME_measurements_.n_RubyPlus_callbacks);
    }

    // +++++ Copy Callbacks +++++ //

    // Copy info (when RB_pc_modificable_callback_ is modified, the thread is blocked)
    TIME_measurements_.time_safety_data_copy.Reset();
    memcpy(&RB_pc_, &RB_pc_modificable_callback_, sizeof(RB_pc_modificable_callback_));

    // We are not waiting for helios... but if they are available take data
    valid_data_HeliosRight_ = false;
    if (consider_PC_HeliosRight_ && flag_HeliosRight_callback_available_)
    {
      memcpy(&Hr_pc_, &Hr_pc_modificable_callback_, sizeof(Hr_pc_modificable_callback_));
      valid_data_HeliosRight_ = true;
    }
    flag_HeliosRight_callback_available_ = false;

    valid_data_HeliosLeft_ = false;
    if (consider_PC_HeliosLeft_ && flag_HeliosLeft_callback_available_)
    {
      memcpy(&Hl_pc_, &Hl_pc_modificable_callback_, sizeof(Hl_pc_modificable_callback_));
      valid_data_HeliosLeft_ = true;
    }
    flag_HeliosLeft_callback_available_ = false;

    TIME_measurements_.time_safety_data_copy.GetElapsedTime();
    TIME_measurements_.time_safety_data_copy.ComputeStats();

    // Copy info ego
    h_info_coche_old_ = h_info_coche_;
    memcpy(&h_info_coche_, &info_coche_modificable_callback_, sizeof(info_coche_modificable_callback_));
    memcpy(&debug_received_info_coche_, &info_coche_modificable_callback_, sizeof(info_coche_modificable_callback_));
    h_info_coche_.largo = 4;
    h_info_coche_.ancho = 2;

    // Reset RubyPlus waiting condition
    flag_avoid_spurious_RubyPlus_callback_ = false;

    // Release thread so callback can continue doing things (only hold it for the shared data management (pc and flag))
    lockRubyPlus.unlock();
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! end lock !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! //

    if (enumeration)
    {
      cudaMemGetInfo(&freeMem, &totalMem);
      printf("\nPoint cloud copied   (memory free = %zu, total = %zu)\n", freeMem, totalMem);
      printf(" - elapsed time = %fms, mean = %fms, max = %fms\n",
             TIME_measurements_.time_safety_data_copy.measured_time, TIME_measurements_.time_safety_data_copy.mean_time,
             TIME_measurements_.time_safety_data_copy.max_time);
    }

    // +++++ Iter timestamp +++++ //
    uint32_t timestamp_iter_sec = RB_pc_.end_sec;
    uint32_t timestamp_iter_nsec = RB_pc_.end_nsec;
    double timestamp_iter = (double)timestamp_iter_sec + ((double)timestamp_iter_nsec) * 1e-9;

    // +++++ Valid Helios? +++++ //
    bool compute_PC_HeliosRight = (consider_PC_HeliosRight_ && valid_data_HeliosRight_) ? true : false;
    bool compute_PC_HeliosLeft = (consider_PC_HeliosLeft_ && valid_data_HeliosLeft_) ? true : false;

    double diff_time_RB_odom =
        ((double)(RB_pc_.end_sec - h_info_coche_.sec)) + ((double)(RB_pc_.end_nsec - h_info_coche_.nanosec)) * 1e-9;
    double diff_time_RB_Hr =
        ((double)(RB_pc_.end_sec - Hr_pc_.end_sec)) + ((double)(RB_pc_.end_nsec - Hr_pc_.end_nsec)) * 1e-9;
    double diff_time_RB_Hl =
        ((double)(RB_pc_.end_sec - Hl_pc_.end_sec)) + ((double)(RB_pc_.end_nsec - Hl_pc_.end_nsec)) * 1e-9;

    printf("Synchronization: \n");

    printf("   RubyPlus (%d)      timestamp: %.10f\n", TIME_measurements_.n_RubyPlus_callbacks,
           ((double)RB_pc_.end_sec) + ((double)RB_pc_.end_nsec) * 1e-9);

    printf("   Odom (%d)         timestamp: %.10f;  diff = %f;\n", TIME_measurements_.n_odom_callbacks,
           h_info_coche_.tiempo, diff_time_RB_odom);

    printf("   H. Right (%d -> %d) timestamp: %.10f;  diff = %f;\n", TIME_measurements_.n_HeliosRight_callbacks,
           (int)compute_PC_HeliosRight, ((double)Hr_pc_.end_sec) + ((double)Hr_pc_.end_nsec) * 1e-9, diff_time_RB_Hr);

    printf("   H. Left  (%d -> %d) timestamp: %.10f;  diff = %f;\n", TIME_measurements_.n_HeliosLeft_callbacks,
           (int)compute_PC_HeliosLeft, ((double)Hl_pc_.end_sec) + ((double)Hl_pc_.end_nsec) * 1e-9, diff_time_RB_Hl);

    bool evitar_prediccion_ego_por_descrincronizacion = false;
    if (std::abs(diff_time_RB_odom) > 0.07)  // Odom 0.02s
    {
      printf(
          "   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CARE. RubyPlus and Localization seem to be unsynchronized "
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

      h_info_coche_.tiempo = timestamp_iter;

      printf(
          "   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CARE. RubyPlus and Localization seem to be unsynchronized "
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
    if (std::abs(diff_time_RB_Hr) > 0.12 && compute_PC_HeliosRight)  // LiDAR 0.1s
    {
      compute_PC_HeliosRight = false;

      printf(
          "   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CARE. RubyPlus and Helios Right seem to be unsynchronized "
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      printf(
          "   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CARE. RubyPlus and Helios Right seem to be unsynchronized "
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      exit(1);
    }
    if (std::abs(diff_time_RB_Hl) > 0.12 && compute_PC_HeliosLeft)
    {
      compute_PC_HeliosLeft = false;

      printf(
          "   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CARE. RubyPlus and Helios Left seem to be "
          "unsynchronized!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      printf(
          "   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CARE. RubyPlus and Helios Left seem to be "
          "unsynchronized!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      exit(1);
    }

    // ++++++++++++++++++++++++++++++++++++++++++ Localization data ++++++++++++++++++++++++++++++++++++++++++ //

    if (publish_tf_odom_world)
    {
      // BEFORE correcting state
      PerceptionObstacles::PublishTransformOdomWorld(h_info_coche_.px_G, h_info_coche_.py_G, h_info_coche_.yaw_G,
                                                     h_info_coche_.sec, h_info_coche_.nanosec);
    }

    // Predecimos el estado coche al instante actual y calculamos el delta pos
    if (evitar_prediccion_ego_por_descrincronizacion == false)
    {
      EGO_VEH::predecir_estado_coche(&h_info_coche_, &h_info_coche_old_, timestamp_iter);
    }
    EGO_VEH::calculo_delta_estado_coche(&h_info_coche_, &h_info_coche_old_, iteration, iter_inicial);
    cudaMemcpy(d_info_coche_, &h_info_coche_, sizeof(EGO_VEH::INFO_ego), cudaMemcpyHostToDevice);

    if (enumeration)
    {
      printf("\nEgo vehicle: (diff ruby = %f)\n", timestamp_iter - debug_received_info_coche_.tiempo);
      // printf("  - received state = [%f, %f, %fº] [%f, %fº] %fs;\n", debug_received_info_coche_.px_G,
      //        debug_received_info_coche_.py_G, debug_received_info_coche_.yaw_G * 180.0 / M_PI,
      //        debug_received_info_coche_.vel, debug_received_info_coche_.yaw_rate * 180.0 / M_PI,
      //        debug_received_info_coche_.tiempo);

      printf("  - current state = [%f, %f, %fº] [%f, %fº] %fs;\n", h_info_coche_.px_G, h_info_coche_.py_G,
             h_info_coche_.yaw_G * 180.0 / M_PI, h_info_coche_.vel, h_info_coche_.yaw_rate * 180.0 / M_PI,
             h_info_coche_.tiempo);

      // printf("  - previous state = [%f, %f, %fº] [%f, %fº] %fs;\n", h_info_coche_old_.px_G,
      h_info_coche_old_.py_G,
          //        h_info_coche_old_.yaw_G * 180.0 / M_PI, h_info_coche_old_.vel, h_info_coche_old_.yaw_rate * 180.0 /
          //        M_PI, h_info_coche_old_.tiempo);

          printf("  -> delta = [%f, %f, %fº, %fs]\n", h_info_coche_.delta_x, h_info_coche_.delta_y,
                 h_info_coche_.delta_yaw * 180.0 / M_PI, h_info_coche_.delta_t);
    }

    // ++++++++++++++++++++++++++++++++++++++++++ Point cloud processing ++++++++++++++++++++++++++++++++++++++++++
    //
    using std::chrono::duration;
    using std::chrono::high_resolution_clock;
    auto t1 = high_resolution_clock::now();
    TIME_measurements_.time_PCs_processing_total.Reset();

    OBST_GROUND::pc_processing_core(&RB_pc_, d_RB_pc_, &param_CB_RB_, compute_PC_HeliosRight, compute_PC_HeliosLeft,
                                    &Hr_pc_, d_Hr_pc_, &Hl_pc_, d_Hl_pc_, &param_CB_Helios_, labels_.label_obst,
                                    labels_.label_suelo, labels_.label_noise, &h_info_coche_, &h_info_coche_old_,
                                    &TIME_measurements_, iteration);

    TIME_measurements_.time_PCs_processing_total.GetElapsedTime();
    TIME_measurements_.time_PCs_processing_total.ComputeStats();
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    if (enumeration)
    {
      cudaMemGetInfo(&freeMem, &totalMem);
      printf("\nPoint cloud classified   (memory free = %zu, total = %zu)\n", freeMem, totalMem);
      printf(" - elapsed time = %fms, mean = %fms, max = %fms (other measurement %f)\n",
             TIME_measurements_.time_PCs_processing_total.measured_time,
             TIME_measurements_.time_PCs_processing_total.mean_time,
             TIME_measurements_.time_PCs_processing_total.max_time, ms_double.count());
    }

    // ++++++++++++++++++++++++++++++++++++++++++ Offline Road Map ++++++++++++++++++++++++++++++++++++++++++

    TIME_measurements_.time_road_map_total.Reset();
    OFF_ROAD_MAP::get_offline_road_map_core(d_grid_road_, d_complete_road_map_, d_config_map_, d_info_coche_,
                                            d_grid_cart_data_);

    TIME_measurements_.time_road_map_total.GetElapsedTime();
    TIME_measurements_.time_road_map_total.ComputeStats();

    if (enumeration)
    {
      cudaMemGetInfo(&freeMem, &totalMem);
      printf("\nOffline road map copied  (memory free = %zu, total = %zu)\n", freeMem, totalMem);
      printf(" - elapsed time = %fms, mean = %fms, max = %fms\n", TIME_measurements_.time_road_map_total.measured_time,
             TIME_measurements_.time_road_map_total.mean_time, TIME_measurements_.time_road_map_total.max_time);
    }

    // ++++++++++++++++++++++++++++++++++++++++++ Observed Occupancy Grid ++++++++++++++++++++++++++++++++++++++++++

    TIME_measurements_.time_obsOG_total.Reset();
    bool consider_OG_HeliosRight = true;
    bool consider_OG_HeliosLeft = true;
    printf("PARAMETROS NO INICIALIZADOS\n");

    bool compute_OG_HeliosRight =
        (consider_OG_HeliosRight && valid_data_HeliosRight_ && compute_PC_HeliosRight) ? true : false;
    bool compute_OG_HeliosLeft =
        (consider_OG_HeliosRight && valid_data_HeliosRight_ && compute_PC_HeliosLeft) ? true : false;

    OBS_OG::compute_observed_occupancy_core(
        d_grid_obs_, h_grid_cart_data_, d_grid_cart_data_, &RB_PolarOG_, d_RB_PolarOG_, &RB_pc_, d_RB_pc_,
        compute_OG_HeliosRight, &Hr_PolarOG_, d_Hr_PolarOG_, &Hr_pc_, d_Hr_pc_, compute_OG_HeliosLeft, &Hl_PolarOG_,
        d_Hl_PolarOG_, &Hl_pc_, d_Hl_pc_, labels_.label_obst, &TIME_measurements_, iteration);

    TIME_measurements_.time_obsOG_total.GetElapsedTime();
    TIME_measurements_.time_obsOG_total.ComputeStats();

    if (enumeration)
    {
      cudaMemGetInfo(&freeMem, &totalMem);
      printf("\nObserved OG computed   (memory free = %zu, total = %zu)\n", freeMem, totalMem);
      printf(" - elapsed time = %fms, mean = %fms, max = %fms\n", TIME_measurements_.time_obsOG_total.measured_time,
             TIME_measurements_.time_obsOG_total.mean_time, TIME_measurements_.time_obsOG_total.max_time);
    }

    printf("                                       estoy ignorando V2X\n");

    // ++++++++++++++++++++++++++++++++++++++++++ Dynamic Occupancy Grid ++++++++++++++++++++++++++++++++++++++++++

    TIME_measurements_.time_DOG_total.Reset();

    DYN_CLASS_OG::compute_dynamic_occupancy_grid_core(
        &flag_particles, d_particles_, d_particles_sorted_, d_particles_for_resampling_, d_grid_, h_config_DOG_,
        d_config_DOG_, d_grid_obs_, h_grid_cart_data_, d_grid_cart_data_, &h_info_coche_, d_info_coche_, d_random_pred,
        d_random_particle_selection, d_random_cell_selection, d_random_asociacion, d_random_vel_uniforme,
        initial_rng_seed + iteration, streams_, &TIME_measurements_, iteration);

    cudaDeviceSynchronize();
    TIME_measurements_.time_DOG_total.GetElapsedTime();
    TIME_measurements_.time_DOG_total.ComputeStats();

    if (enumeration)
    {
      cudaMemGetInfo(&freeMem, &totalMem);
      printf("\nDOG computed   (memory free = %zu, total = %zu)\n", freeMem, totalMem);
      printf(" - elapsed time = %fms, mean = %fms, max = %fms\n", TIME_measurements_.time_DOG_total.measured_time,
             TIME_measurements_.time_DOG_total.mean_time, TIME_measurements_.time_DOG_total.max_time);
    }

    // ------------------------------ DEBUG ------------------------------ //
    // Publish
    if (true)
    {
      static ChronoTimer time_publication;
      time_publication.Reset();

      checkCudaErrors(cudaMemcpy(&RB_pc_, d_RB_pc_, sizeof(AUTOPIA_RubyPlus::PointCloud), cudaMemcpyDeviceToHost));
      // cudaDeviceSynchronize(); Not needed because cudamemcpy in the main stream already blocks and waits
      pc_RB_publisher_->publish(PerceptionObstacles::GeneratePointCloud2Message(
          RB_pc_.x, RB_pc_.y, RB_pc_.z, RB_pc_.intensity, RB_pc_.vert_ang, RB_pc_.label, RB_pc_.channel_label,
          RB_pc_.metaChannel, RB_pc_.n_points, "odom", timestamp_iter_sec, timestamp_iter_nsec));

      if (compute_PC_HeliosRight)
      {
        checkCudaErrors(cudaMemcpy(&Hr_pc_, d_Hr_pc_, sizeof(AUTOPIA_Helios::PointCloud), cudaMemcpyDeviceToHost));
        // cudaDeviceSynchronize(); Not needed because cudamemcpy in the main stream already blocks and waits
        pc_Hr_publisher_->publish(PerceptionObstacles::GeneratePointCloud2Message(
            Hr_pc_.x, Hr_pc_.y, Hr_pc_.z, Hr_pc_.intensity, Hr_pc_.vert_ang, Hr_pc_.label, Hr_pc_.channel_label,
            Hr_pc_.metaChannel, Hr_pc_.n_points, "odom", timestamp_iter_sec, timestamp_iter_nsec));
      }

      if (compute_PC_HeliosLeft)
      {
        checkCudaErrors(cudaMemcpy(&Hl_pc_, d_Hl_pc_, sizeof(AUTOPIA_Helios::PointCloud), cudaMemcpyDeviceToHost));
        // cudaDeviceSynchronize(); Not needed because cudamemcpy in the main stream already blocks and waits
        pc_Hl_publisher_->publish(PerceptionObstacles::GeneratePointCloud2Message(
            Hl_pc_.x, Hl_pc_.y, Hl_pc_.z, Hl_pc_.intensity, Hl_pc_.vert_ang, Hl_pc_.label, Hl_pc_.channel_label,
            Hl_pc_.metaChannel, Hl_pc_.n_points, "odom", timestamp_iter_sec, timestamp_iter_nsec));
      }

      time_publication.GetElapsedTime();
      time_publication.ComputeStats();
      printf("\nPublish PC\n");
      printf(" - elapsed time = %fms, mean = %fms, max = %fms\n", time_publication.measured_time,
             time_publication.mean_time, time_publication.max_time);
    }

    if (true)
    {
      static ChronoTimer time_publication;
      time_publication.Reset();
      checkCudaErrors(cudaMemcpy(h_grid_obs_, d_grid_obs_, sizeof(GRID_TYPES::OG), cudaMemcpyDeviceToHost));
      // cudaDeviceSynchronize(); Not needed because cudamemcpy in the main stream already blocks and waits

      nav_msgs::msg::OccupancyGrid obs_pO_msg = PerceptionObstacles::GenerateOGMessage(
          h_grid_obs_->pO, GRID_TYPES::NC_X, GRID_TYPES::NC_Y, h_grid_cart_data_->MIN_X, h_grid_cart_data_->MIN_Y,
          GRID_TYPES::RES, timestamp_iter_sec, timestamp_iter_nsec);
      obs_OG_pO_publisher_->publish(obs_pO_msg);

      time_publication.GetElapsedTime();
      time_publication.ComputeStats();
      printf("\nPublish observed OG\n");
      printf(" - elapsed time = %fms, mean = %fms, max = %fms\n", time_publication.measured_time,
             time_publication.mean_time, time_publication.max_time);
    }

    if (false)
    {
      checkCudaErrors(cudaMemcpy(&RB_pc_, d_RB_pc_, sizeof(AUTOPIA_RubyPlus::PointCloud), cudaMemcpyDeviceToHost));
      // cudaDeviceSynchronize(); Not needed because cudamemcpy in the main stream already blocks and waits
      std::string name = "RB_classified_" + std::to_string(iteration);
      OBST_GROUND::write_file_pointcloud(name, RB_pc_.x, RB_pc_.y, RB_pc_.z, RB_pc_.vert_ang, RB_pc_.intensity,
                                         RB_pc_.metaChannel, RB_pc_.label, RB_pc_.channel_label, RB_pc_.label_reason,
                                         RB_pc_.n_points);

      if (compute_PC_HeliosRight)
      {
        checkCudaErrors(cudaMemcpy(&Hr_pc_, d_Hr_pc_, sizeof(AUTOPIA_Helios::PointCloud), cudaMemcpyDeviceToHost));
        // cudaDeviceSynchronize(); Not needed because cudamemcpy in the main stream already blocks and waits
        name = "Hr_classified_" + std::to_string(iteration);
        OBST_GROUND::write_file_pointcloud(name, Hr_pc_.x, Hr_pc_.y, Hr_pc_.z, Hr_pc_.vert_ang, Hr_pc_.intensity,
                                           Hr_pc_.metaChannel, Hr_pc_.label, Hr_pc_.channel_label, Hr_pc_.label_reason,
                                           Hr_pc_.n_points);
      }

      if (compute_PC_HeliosLeft)
      {
        checkCudaErrors(cudaMemcpy(&Hl_pc_, d_Hl_pc_, sizeof(AUTOPIA_Helios::PointCloud), cudaMemcpyDeviceToHost));
        // cudaDeviceSynchronize(); Not needed because cudamemcpy in the main stream already blocks and waits
        name = "Hl_classified_" + std::to_string(iteration);
        OBST_GROUND::write_file_pointcloud(name, Hl_pc_.x, Hl_pc_.y, Hl_pc_.z, Hl_pc_.vert_ang, Hl_pc_.intensity,
                                           Hl_pc_.metaChannel, Hl_pc_.label, Hl_pc_.channel_label, Hl_pc_.label_reason,
                                           Hl_pc_.n_points);
      }
    }

    if (true)
    {
      OBS_OG::write_files_observed_occupancy_grid(d_grid_obs_->mO, d_grid_obs_->mF, iteration);
    }

    if (true)
    {
      DYN_CLASS_OG::write_files_DOG_color(d_grid_, iteration, h_config_DOG_);
      DYN_CLASS_OG::write_files_num_particles(d_grid_, iteration, h_config_DOG_);
    }

    write_files_time(&TIME_measurements_, false);
    EGO_VEH::write_files_localization(&debug_received_info_coche_, false);

    // Medición de tiempo total de ejecución
    TIME_measurements_.time_total_iteration.GetElapsedTime();
    TIME_measurements_.time_total_iteration.ComputeStats();
    printf("\n --------------- Total run time = %fms, mean = %fms, max = %fms --------------- \n\n\n",
           TIME_measurements_.time_total_iteration.measured_time, TIME_measurements_.time_total_iteration.mean_time,
           TIME_measurements_.time_total_iteration.max_time);

    ++iteration;
  }
}

/** ----------------------------------------------------------------------------------------------------------------<
 * @brief main
 */
int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PerceptionObstacles>();

  rclcpp::spin(node);
  rclcpp::shutdown();

  /*
  // initialization of ROS and DDS middleware
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PerceptionObstacles>();

  rclcpp::executors::SingleThreadedExecutor executor;

  rclcpp::executors::SingleThreadedExecutor real_time_executor;
  real_time_executor.add_node(node);

  std::thread thr;

  thr = std::thread([&real_time_executor] {
    sched_param sch;
    sch.sched_priority = 90;
    if (sched_setscheduler(0, SCHED_FIFO, &sch) == -1)
    {
      throw std::runtime_error{ std::string("failed to set scheduler: ") + std::strerror(errno) };
    }
    real_time_executor.spin();
  });

  executor.spin();
  rclcpp::shutdown();

  thr.join();
  */

  return 0;
}