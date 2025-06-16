#include "rclcpp/rclcpp.hpp"
#include "yolo_custom_interfaces/msg/instance_segmentation_info.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <fstream>   // Para std::ofstream
#include <iomanip>   // Para std::fixed, std::setprecision
#include <vector>    // Para std::vector (aunque no se usa directamente, es común)
#include <map>       // Para std::map
#include <string>    // Para std::string
#include <limits>    // Para std::numeric_limits
#include <cmath>     // Para std::sqrt (si se quisiera desviación estándar)
#include <filesystem>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread> // Para std::thread
#include <queue>  // Para std::queue
#include <condition_variable> // Para std::condition_variable
#include <atomic> // Para std::atomic
#include <sstream> // Para std::ostringstream
#include <utility> // Para std::pair

using namespace std::chrono_literals;

// Estructura para calcular y almacenar estadísticas de latencia
struct LatencyMetrics
{
    long count = 0;
    double sum_ms = 0.0;
    double sum_sq_ms = 0.0; // Suma de cuadrados para la varianza
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = std::numeric_limits<double>::min();
    double mean_ms = 0.0;
    double variance_ms = 0.0;

    void update(double value_ms)
    {
        count++;
        sum_ms += value_ms;
        sum_sq_ms += value_ms * value_ms;
        mean_ms = sum_ms / count;

        if (count > 1)
        {
            variance_ms = (sum_sq_ms - (sum_ms * sum_ms) / count) / (count - 1);
        }
        else
        {
            variance_ms = 0.0; // Varianza no definida para menos de 2 muestras
        }

        if (value_ms < min_ms)
            min_ms = value_ms;
        if (value_ms > max_ms)
            max_ms = value_ms;
    }

    void reset()
    {
        count = 0;
        sum_ms = 0.0;
        sum_sq_ms = 0.0;
        min_ms = std::numeric_limits<double>::max();
        max_ms = std::numeric_limits<double>::min();
        mean_ms = 0.0;
        variance_ms = 0.0;
    }
};

static bool timespec_less_than(const timespec& a, const timespec& b) {
    if (a.tv_sec != b.tv_sec) return a.tv_sec < b.tv_sec;
    return a.tv_nsec < b.tv_nsec;
}


class FrequencySubscriber : public rclcpp::Node
{
public:
    FrequencySubscriber()
        : Node("result_frequency_subscriber"), stop_csv_writer_thread_(false)
    {
        // Inicializar thread que guarda las latencias en un archivo CSV

        csv_writer_thread_ = std::thread(&FrequencySubscriber::csvWriterLoop, this);

        // Abrir archivo CSV para escritura
        initialize_colors(); 

        auto callback =
            [this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg, const std::string &camera_id)
        {
            // Initialize tracking for this camera if it's the first time
            initialize_camera_loss_tracking(camera_id);
            timespec ts_t4_segsub_res_recv; // T4 - SegSub Result Reception Time
            clock_gettime(CLOCK_MONOTONIC, &ts_t4_segsub_res_recv);

            timespec ts_t0_img_pub; // T0 - Original Image Publication Time (from publisher's CLOCK_MONOTONIC)
            ts_t0_img_pub.tv_sec = msg->header.stamp.sec;
            ts_t0_img_pub.tv_nsec = msg->header.stamp.nanosec;

            timespec ts_t1_segnode_cb_entry; // T1 - SegNode Image Callback Entry Time
            ts_t1_segnode_cb_entry.tv_sec = msg->image_source_monotonic_capture_time.sec;
            ts_t1_segnode_cb_entry.tv_nsec = msg->image_source_monotonic_capture_time.nanosec;

            timespec ts_t2_segnode_batch_start; // T2 - SegNode Batch Processing Start Time
            ts_t2_segnode_batch_start.tv_sec = msg->processing_node_monotonic_entry_time.sec;
            ts_t2_segnode_batch_start.tv_nsec = msg->processing_node_monotonic_entry_time.nanosec;

            // --- Extract Inference Timestamps ---
            timespec ts_t2a_inference_start; // T2a
            ts_t2a_inference_start.tv_sec = msg->processing_node_inference_start_time.sec;
            ts_t2a_inference_start.tv_nsec = msg->processing_node_inference_start_time.nanosec;

            timespec ts_t2b_inference_end; // T2b
            ts_t2b_inference_end.tv_sec = msg->processing_node_inference_end_time.sec;
            ts_t2b_inference_end.tv_nsec = msg->processing_node_inference_end_time.nanosec;
            // --- End Inference Timestamps ---

            timespec ts_t3_segnode_res_pub; // T3 - SegNode Result Publish Time
            ts_t3_segnode_res_pub.tv_sec = msg->processing_node_monotonic_publish_time.sec;
            ts_t3_segnode_res_pub.tv_nsec = msg->processing_node_monotonic_publish_time.nanosec;


            uint64_t packet_seq_num = msg->packet_sequence_number;

            auto calculate_latency_ms_local = [](const timespec& t_end, const timespec& t_start) {
                if ((t_start.tv_sec == 0 && t_start.tv_nsec == 0) || (t_end.tv_sec == 0 && t_end.tv_nsec == 0)) { 
                    return std::numeric_limits<double>::quiet_NaN();
                }
                if (t_end.tv_sec < t_start.tv_sec || (t_end.tv_sec == t_start.tv_sec && t_end.tv_nsec < t_start.tv_nsec)) {
                    // Negative latency, indicate error or issue
                    // For now, return NaN, or a specific error code if preferred
                    return std::numeric_limits<double>::quiet_NaN(); 
                }
                return (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                       (static_cast<double>(t_end.tv_nsec) - static_cast<double>(t_start.tv_nsec)) / 1e6;
            };
            
            // Calculate individual latencies
            double lat_t0_t1_ms = calculate_latency_ms_local(ts_t1_segnode_cb_entry, ts_t0_img_pub);
            double lat_t1_t2_ms = calculate_latency_ms_local(ts_t2_segnode_batch_start, ts_t1_segnode_cb_entry);
            double lat_t2_t2a_ms = calculate_latency_ms_local(ts_t2a_inference_start, ts_t2_segnode_batch_start);
            double lat_t2a_t2b_ms = calculate_latency_ms_local(ts_t2b_inference_end, ts_t2a_inference_start); // Inference Duration
            double lat_t2b_t3_ms = calculate_latency_ms_local(ts_t3_segnode_res_pub, ts_t2b_inference_end);
            double lat_t3_t4_ms = calculate_latency_ms_local(ts_t4_segsub_res_recv, ts_t3_segnode_res_pub);
            
            double lat_total_e2e_ms = calculate_latency_ms_local(ts_t4_segsub_res_recv, ts_t0_img_pub);
            double lat_segnode_cb_to_segsub_recv_ms = calculate_latency_ms_local(ts_t4_segsub_res_recv, ts_t1_segnode_cb_entry);

            // Variables for spreads
            double t0_spread_ms = std::numeric_limits<double>::quiet_NaN();
            double t1_spread_ms = std::numeric_limits<double>::quiet_NaN();
            double t2_spread_ms = std::numeric_limits<double>::quiet_NaN();
            double t2a_spread_ms = std::numeric_limits<double>::quiet_NaN();
            double t2b_spread_ms = std::numeric_limits<double>::quiet_NaN();
            double t3_spread_ms = std::numeric_limits<double>::quiet_NaN();
            double t4_spread_ms = std::numeric_limits<double>::quiet_NaN();

            bool t0_spread_calculated_this_call = false;
            bool t1_spread_calculated_this_call = false;
            bool t2_spread_calculated_this_call = false;
            bool t2a_spread_calculated_this_call = false;
            bool t2b_spread_calculated_this_call = false;
            bool t3_spread_calculated_this_call = false;
            bool t4_spread_calculated_this_call = false;


            // Variables for parallelism metrics
            double sum_individual_inf_dur_ms = std::numeric_limits<double>::quiet_NaN();
            double total_batch_inf_span_ms = std::numeric_limits<double>::quiet_NaN();
            double inf_overlap_time_ms = std::numeric_limits<double>::quiet_NaN();
            double parallel_overlap_pct = std::numeric_limits<double>::quiet_NaN();
            double inf_concurrency_factor = std::numeric_limits<double>::quiet_NaN();
            bool parallelism_metrics_calculated_this_call = false;

            {
                std::lock_guard<std::mutex> lock(batch_spread_data_mutex_);

                // Store current timestamps
                received_batch_t0_timestamps_[packet_seq_num][camera_id] = ts_t0_img_pub;
                received_batch_t1_timestamps_[packet_seq_num][camera_id] = ts_t1_segnode_cb_entry;
                received_batch_t2_timestamps_[packet_seq_num][camera_id] = ts_t2_segnode_batch_start;
                received_batch_t2a_timestamps_[packet_seq_num][camera_id] = ts_t2a_inference_start;
                received_batch_t2b_timestamps_[packet_seq_num][camera_id] = ts_t2b_inference_end;
                received_batch_t3_timestamps_[packet_seq_num][camera_id] = ts_t3_segnode_res_pub;
                received_batch_t4_timestamps_[packet_seq_num][camera_id] = ts_t4_segsub_res_recv;

                // Helper lambda to calculate spread for a given stage
                auto calculate_stage_spread = 
                    [&](uint64_t p_seq_num, 
                        std::map<uint64_t, std::map<std::string, timespec>>& received_map,
                        std::map<uint64_t, double>& calculated_map,
                        double& out_spread_value, bool& out_calculated_this_call) {
                    if (calculated_map.find(p_seq_num) == calculated_map.end()) {
                        if (received_map[p_seq_num].count("left") &&
                            received_map[p_seq_num].count("front") &&
                            received_map[p_seq_num].count("right")) {
                            std::vector<timespec> t_stamps = {
                                received_map[p_seq_num]["left"],
                                received_map[p_seq_num]["front"],
                                received_map[p_seq_num]["right"]
                            };
                            auto min_t_it = std::min_element(t_stamps.begin(), t_stamps.end(), timespec_less_than);
                            auto max_t_it = std::max_element(t_stamps.begin(), t_stamps.end(), timespec_less_than);
                            if (min_t_it != t_stamps.end() && max_t_it != t_stamps.end()) {
                                out_spread_value = calculate_latency_ms_local(*max_t_it, *min_t_it);
                                calculated_map[p_seq_num] = out_spread_value;
                                out_calculated_this_call = true;
                            }
                        }
                    } else {
                        out_spread_value = calculated_map[p_seq_num];
                    }
                };

                calculate_stage_spread(packet_seq_num, received_batch_t0_timestamps_, batch_t0_spread_ms_, t0_spread_ms, t0_spread_calculated_this_call);
                calculate_stage_spread(packet_seq_num, received_batch_t1_timestamps_, batch_t1_spread_ms_, t1_spread_ms, t1_spread_calculated_this_call);
                calculate_stage_spread(packet_seq_num, received_batch_t2_timestamps_, batch_t2_spread_ms_, t2_spread_ms, t2_spread_calculated_this_call);
                calculate_stage_spread(packet_seq_num, received_batch_t2a_timestamps_, batch_t2a_spread_ms_, t2a_spread_ms, t2a_spread_calculated_this_call);
                calculate_stage_spread(packet_seq_num, received_batch_t2b_timestamps_, batch_t2b_spread_ms_, t2b_spread_ms, t2b_spread_calculated_this_call);
                calculate_stage_spread(packet_seq_num, received_batch_t3_timestamps_, batch_t3_spread_ms_, t3_spread_ms, t3_spread_calculated_this_call);
                calculate_stage_spread(packet_seq_num, received_batch_t4_timestamps_, batch_t4_spread_ms_, t4_spread_ms, t4_spread_calculated_this_call);
            }

            // Calculate Parallelism Metrics
            if (batch_parallelism_metrics_calculated_.find(packet_seq_num) == batch_parallelism_metrics_calculated_.end()) {
                if (received_batch_t2a_timestamps_[packet_seq_num].count("left") && received_batch_t2a_timestamps_[packet_seq_num].count("front") && received_batch_t2a_timestamps_[packet_seq_num].count("right") &&
                    received_batch_t2b_timestamps_[packet_seq_num].count("left") && received_batch_t2b_timestamps_[packet_seq_num].count("front") && received_batch_t2b_timestamps_[packet_seq_num].count("right")) {
                    
                    timespec min_t2a_batch = received_batch_t2a_timestamps_[packet_seq_num]["left"];
                    if (timespec_less_than(received_batch_t2a_timestamps_[packet_seq_num]["front"], min_t2a_batch)) min_t2a_batch = received_batch_t2a_timestamps_[packet_seq_num]["front"];
                    if (timespec_less_than(received_batch_t2a_timestamps_[packet_seq_num]["right"], min_t2a_batch)) min_t2a_batch = received_batch_t2a_timestamps_[packet_seq_num]["right"];

                    timespec max_t2b_batch = received_batch_t2b_timestamps_[packet_seq_num]["left"];
                    if (timespec_less_than(max_t2b_batch, received_batch_t2b_timestamps_[packet_seq_num]["front"])) max_t2b_batch = received_batch_t2b_timestamps_[packet_seq_num]["front"];
                    if (timespec_less_than(max_t2b_batch, received_batch_t2b_timestamps_[packet_seq_num]["right"])) max_t2b_batch = received_batch_t2b_timestamps_[packet_seq_num]["right"];
                    
                    total_batch_inf_span_ms = calculate_latency_ms_local(max_t2b_batch, min_t2a_batch);

                    double lat_inf_left = calculate_latency_ms_local(received_batch_t2b_timestamps_[packet_seq_num]["left"], received_batch_t2a_timestamps_[packet_seq_num]["left"]);
                    double lat_inf_front = calculate_latency_ms_local(received_batch_t2b_timestamps_[packet_seq_num]["front"], received_batch_t2a_timestamps_[packet_seq_num]["front"]);
                    double lat_inf_right = calculate_latency_ms_local(received_batch_t2b_timestamps_[packet_seq_num]["right"], received_batch_t2a_timestamps_[packet_seq_num]["right"]);

                    sum_individual_inf_dur_ms = 0;
                    if (!std::isnan(lat_inf_left)) sum_individual_inf_dur_ms += lat_inf_left;
                    if (!std::isnan(lat_inf_front)) sum_individual_inf_dur_ms += lat_inf_front;
                    if (!std::isnan(lat_inf_right)) sum_individual_inf_dur_ms += lat_inf_right;
                    
                    if (!std::isnan(sum_individual_inf_dur_ms) && !std::isnan(total_batch_inf_span_ms)) {
                        if (total_batch_inf_span_ms > 1e-9) { // Avoid division by zero or tiny numbers
                            inf_concurrency_factor = sum_individual_inf_dur_ms / total_batch_inf_span_ms;
                        } else if (sum_individual_inf_dur_ms > 1e-9) { // Span is zero but sum is not, implies infinite concurrency (or error)
                            inf_concurrency_factor = std::numeric_limits<double>::infinity();
                        } else { // Both zero
                            inf_concurrency_factor = 1.0; // Or NaN, define as 1 for no work done
                        }

                        inf_overlap_time_ms = sum_individual_inf_dur_ms - total_batch_inf_span_ms;
                        if (sum_individual_inf_dur_ms > 1e-9) {
                            parallel_overlap_pct = (inf_overlap_time_ms / sum_individual_inf_dur_ms) * 100.0;
                            parallel_overlap_pct = std::max(0.0, std::min(100.0, parallel_overlap_pct)); // Clamp
                        } else {
                            parallel_overlap_pct = 0.0; // No work, so 0% overlap
                        }
                    }
                    
                    // Store calculated batch metrics
                    batch_sum_individual_inf_dur_ms_[packet_seq_num] = sum_individual_inf_dur_ms;
                    batch_total_inf_span_ms_[packet_seq_num] = total_batch_inf_span_ms;
                    batch_inf_overlap_time_ms_[packet_seq_num] = inf_overlap_time_ms;
                    batch_parallel_overlap_pct_[packet_seq_num] = parallel_overlap_pct;
                    batch_inf_concurrency_factor_[packet_seq_num] = inf_concurrency_factor;
                    batch_parallelism_metrics_calculated_[packet_seq_num] = true;
                    parallelism_metrics_calculated_this_call = true;
                }
            } else { // Already calculated for this batch, retrieve them
                sum_individual_inf_dur_ms = batch_sum_individual_inf_dur_ms_[packet_seq_num];
                total_batch_inf_span_ms = batch_total_inf_span_ms_[packet_seq_num];
                inf_overlap_time_ms = batch_inf_overlap_time_ms_[packet_seq_num];
                parallel_overlap_pct = batch_parallel_overlap_pct_[packet_seq_num];
                inf_concurrency_factor = batch_inf_concurrency_factor_[packet_seq_num];
            }
            
            
            // T0 offsets (already implemented, can be kept or removed if T0 spread is preferred)
            double offset_T0_front_vs_left_ms = std::numeric_limits<double>::quiet_NaN();
            double offset_T0_right_vs_left_ms = std::numeric_limits<double>::quiet_NaN();
            bool batch_offsets_calculated_this_call = false; // This logic might need review or integration with T0 spread

            // ... (loss tracking logic remains the same) ...
            uint64_t current_seq_num = msg->packet_sequence_number; 
            uint64_t lost_count_since_last = 0;
            { // Alcance para el lock de seguimiento de pérdidas
                std::lock_guard<std::mutex> lock(loss_tracking_mutex_);
                
                if (!first_packet_received_flag_[camera_id]) { 
                    last_received_seq_num_[camera_id] = current_seq_num;
                    first_packet_received_flag_[camera_id] = true;
                } else {
                    if (current_seq_num == last_received_seq_num_[camera_id] + 1) {
                        // No hay paquetes perdidos
                    } 
                    else if (current_seq_num > last_received_seq_num_[camera_id] + 1) {
                        lost_count_since_last = current_seq_num - (last_received_seq_num_[camera_id] + 1);
                        lost_packets_total_count_[camera_id] += lost_count_since_last;
                        RCLCPP_WARN(this->get_logger(), "[%s Pkt#%lu] Lost %lu packets. Expected seq %lu. Total lost for cam: %lu",
                                    camera_id.c_str(), current_seq_num, lost_count_since_last, last_received_seq_num_[camera_id] + 1, lost_packets_total_count_[camera_id]);
                    }
                    else if (current_seq_num < last_received_seq_num_[camera_id] +1) {
                         RCLCPP_WARN(this->get_logger(), "[%s Pkt#%lu] Received out-of-order or reset sequence. Last was %lu. Ignoring for loss count.",
                                    camera_id.c_str(), current_seq_num, last_received_seq_num_[camera_id]);
                    }
                    last_received_seq_num_[camera_id] = current_seq_num; 
                }
            } 

            // Log visual
            std::ostringstream log_stream;
            log_stream << std::fixed << std::setprecision(3);
            log_stream << "\n======================================================================================\n";
            log_stream << "[" << camera_id << " Pkt#" << packet_seq_num << "] Latency Breakdown. ImgPub OriginalTS: "
                       << static_cast<long>(msg->header.stamp.sec) << "." << std::setfill('0') << std::setw(9) << static_cast<unsigned long>(msg->header.stamp.nanosec) << "\n";
            log_stream << "--------------------------------[ TIMESTAMPS (MONOTONIC) ]------------------------------\n";
            log_stream << "  T0_ImgPubTime ....................: " << ts_t0_img_pub.tv_sec << "." << std::setfill('0') << std::setw(9) << ts_t0_img_pub.tv_nsec << "\n";
            log_stream << "  T1_SegNode_CBCallTime ............: " << ts_t1_segnode_cb_entry.tv_sec << "." << std::setfill('0') << std::setw(9) << ts_t1_segnode_cb_entry.tv_nsec << "\n";
            log_stream << "  T2_SegNode_BatchProcStartTime ....: " << ts_t2_segnode_batch_start.tv_sec << "." << std::setfill('0') << std::setw(9) << ts_t2_segnode_batch_start.tv_nsec << "\n";
            log_stream << "  T2a_SegNode_InferenceStartTime ...: " << ts_t2a_inference_start.tv_sec << "." << std::setfill('0') << std::setw(9) << ts_t2a_inference_start.tv_nsec << "\n";
            log_stream << "  T2b_SegNode_InferenceEndTime .....: " << ts_t2b_inference_end.tv_sec << "." << std::setfill('0') << std::setw(9) << ts_t2b_inference_end.tv_nsec << "\n";
            log_stream << "  T3_SegNode_ResultPubTime .........: " << ts_t3_segnode_res_pub.tv_sec << "." << std::setfill('0') << std::setw(9) << ts_t3_segnode_res_pub.tv_nsec << "\n";
            log_stream << "  T4_SegSub_ResultRecvTime .........: " << ts_t4_segsub_res_recv.tv_sec << "." << std::setfill('0') << std::setw(9) << ts_t4_segsub_res_recv.tv_nsec << "\n";
            log_stream << "-----------------------------------[ LATENCIES (ms) ]-----------------------------------\n";
            log_stream << "  T0->T1 (ImgPub to CB).............: " << (std::isnan(lat_t0_t1_ms) ? "NaN" : std::to_string(lat_t0_t1_ms)) << "\n";
            log_stream << "  T1->T2 (CB to BatchStart).........: " << (std::isnan(lat_t1_t2_ms) ? "NaN" : std::to_string(lat_t1_t2_ms)) << "\n";
            log_stream << "  T2->T2a (BatchStart to InfStart)..: " << (std::isnan(lat_t2_t2a_ms) ? "NaN" : std::to_string(lat_t2_t2a_ms)) << "\n";
            log_stream << "  T2a->T2b (Inference Duration).....: " << (std::isnan(lat_t2a_t2b_ms) ? "NaN" : std::to_string(lat_t2a_t2b_ms)) << "\n";
            log_stream << "  T2b->T3 (InfEnd to ResPub)........: " << (std::isnan(lat_t2b_t3_ms) ? "NaN" : std::to_string(lat_t2b_t3_ms)) << "\n";
            log_stream << "  T3->T4 (ResPub to SegSubRecv).....: " << (std::isnan(lat_t3_t4_ms) ? "NaN" : std::to_string(lat_t3_t4_ms)) << "\n";
            log_stream << "-----------------------------[ CUMULATIVE LATENCIES (ms) ]------------------------------\n";
            log_stream << "  SegNodeCB to SegSubRecv (T1->T4)..: " << (std::isnan(lat_segnode_cb_to_segsub_recv_ms) ? "NaN" : std::to_string(lat_segnode_cb_to_segsub_recv_ms)) << "\n";
            log_stream << "  TOTAL E2E (ImgPub to SegSubRecv) .: " << (std::isnan(lat_total_e2e_ms) ? "NaN" : std::to_string(lat_total_e2e_ms)) << "\n";
            log_stream << "-----------------------------[ BATCH SPREADS (ms) Pkt#" << packet_seq_num << " ]--------------------------\n";
            log_stream << "  T0 Spread (Publisher).............: " << (std::isnan(t0_spread_ms) ? "N/A" : std::to_string(t0_spread_ms)) << "\n";
            log_stream << "  T1 Spread (SegNode CB Entry)......: " << (std::isnan(t1_spread_ms) ? "N/A" : std::to_string(t1_spread_ms)) << "\n";
            log_stream << "  T2 Spread (SegNode BatchStart)....: " << (std::isnan(t2_spread_ms) ? "N/A" : std::to_string(t2_spread_ms)) << "\n";
            log_stream << "  T2a Spread (SegNode InfStart)....: " << (std::isnan(t2a_spread_ms) ? "N/A" : std::to_string(t2a_spread_ms)) << "\n";
            log_stream << "  T2b Spread (SegNode InfEnd).......: " << (std::isnan(t2b_spread_ms) ? "N/A" : std::to_string(t2b_spread_ms)) << "\n";
            log_stream << "  T3 Spread (SegNode ResPub)........: " << (std::isnan(t3_spread_ms) ? "N/A" : std::to_string(t3_spread_ms)) << "\n";
            log_stream << "  T4 Spread (SegSub Recv)...........: " << (std::isnan(t4_spread_ms) ? "N/A" : std::to_string(t4_spread_ms)) << "\n";
            log_stream << "-------------------------[ INFERENCE PARALLELISM (ms) Pkt#" << packet_seq_num << " ]----------------------\n";
            log_stream << "  Sum Individual Inf Durations (S)..: " << (std::isnan(sum_individual_inf_dur_ms) ? "N/A" : std::to_string(sum_individual_inf_dur_ms)) << "\n";
            log_stream << "  Total Batch Inf Span (P)..........: " << (std::isnan(total_batch_inf_span_ms) ? "N/A" : std::to_string(total_batch_inf_span_ms)) << "\n";
            log_stream << "  Inference Overlap Time (S-P)......: " << (std::isnan(inf_overlap_time_ms) ? "N/A" : std::to_string(inf_overlap_time_ms)) << "\n";
            log_stream << "  Parallel Overlap Pct ((S-P)/S)...: " << (std::isnan(parallel_overlap_pct) ? "N/A" : std::to_string(parallel_overlap_pct)) << " %\n";
            log_stream << "  Inference Concurrency (S/P).......: " << (std::isnan(inf_concurrency_factor) ? "N/A" : std::to_string(inf_concurrency_factor)) << "\n";
            log_stream << "======================================================================================\n";
            RCLCPP_INFO(this->get_logger(), "%s", log_stream.str().c_str());

            std::ostringstream csv_line_stream;
            csv_line_stream << std::fixed << std::setprecision(3)
                            << camera_id << ","
                            << packet_seq_num << "," 
                            << msg->header.stamp.nanosec << "," 
                            << ts_t0_img_pub.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t0_img_pub.tv_nsec << ","
                            << ts_t1_segnode_cb_entry.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t1_segnode_cb_entry.tv_nsec << ","
                            << ts_t2_segnode_batch_start.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t2_segnode_batch_start.tv_nsec << ","
                            << ts_t2a_inference_start.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t2a_inference_start.tv_nsec << ","
                            << ts_t2b_inference_end.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t2b_inference_end.tv_nsec << ","
                            << ts_t3_segnode_res_pub.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t3_segnode_res_pub.tv_nsec << ","
                            << ts_t4_segsub_res_recv.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t4_segsub_res_recv.tv_nsec << ","
                            << (std::isnan(lat_t0_t1_ms) ? "" : std::to_string(lat_t0_t1_ms)) << ","
                            << (std::isnan(lat_t1_t2_ms) ? "" : std::to_string(lat_t1_t2_ms)) << ","
                            << (std::isnan(lat_t2_t2a_ms) ? "" : std::to_string(lat_t2_t2a_ms)) << ","
                            << (std::isnan(lat_t2a_t2b_ms) ? "" : std::to_string(lat_t2a_t2b_ms)) << ","
                            << (std::isnan(lat_t2b_t3_ms) ? "" : std::to_string(lat_t2b_t3_ms)) << ","
                            << (std::isnan(lat_t3_t4_ms) ? "" : std::to_string(lat_t3_t4_ms)) << ","
                            << (std::isnan(lat_segnode_cb_to_segsub_recv_ms) ? "" : std::to_string(lat_segnode_cb_to_segsub_recv_ms)) << ","
                            << (std::isnan(lat_total_e2e_ms) ? "" : std::to_string(lat_total_e2e_ms));

            // Spreads for CSV
            csv_line_stream << "," << (std::isnan(t0_spread_ms) ? "" : std::to_string(t0_spread_ms))
                            << "," << (std::isnan(t1_spread_ms) ? "" : std::to_string(t1_spread_ms))
                            << "," << (std::isnan(t2_spread_ms) ? "" : std::to_string(t2_spread_ms))
                            << "," << (std::isnan(t2a_spread_ms) ? "" : std::to_string(t2a_spread_ms))
                            << "," << (std::isnan(t2b_spread_ms) ? "" : std::to_string(t2b_spread_ms))
                            << "," << (std::isnan(t3_spread_ms) ? "" : std::to_string(t3_spread_ms))
                            << "," << (std::isnan(t4_spread_ms) ? "" : std::to_string(t4_spread_ms));


            // Parallelism metrics for CSV
            csv_line_stream << "," << (std::isnan(sum_individual_inf_dur_ms) ? "" : std::to_string(sum_individual_inf_dur_ms))
                            << "," << (std::isnan(total_batch_inf_span_ms) ? "" : std::to_string(total_batch_inf_span_ms))
                            << "," << (std::isnan(inf_overlap_time_ms) ? "" : std::to_string(inf_overlap_time_ms))
                            << "," << (std::isnan(parallel_overlap_pct) ? "" : std::to_string(parallel_overlap_pct))
                            << "," << (std::isnan(inf_concurrency_factor) ? "" : std::to_string(inf_concurrency_factor));
            
            // T0 offsets (keeping for now, can be removed if T0 spread is sufficient)
            csv_line_stream << ","; 
            if (!std::isnan(offset_T0_front_vs_left_ms)) csv_line_stream << offset_T0_front_vs_left_ms;
            csv_line_stream << ","; 
            if (!std::isnan(offset_T0_right_vs_left_ms)) csv_line_stream << offset_T0_right_vs_left_ms;

            uint64_t current_total_lost = 0;
            {
                std::lock_guard<std::mutex> lock(loss_tracking_mutex_);
                if(lost_packets_total_count_.count(camera_id)) { 
                    current_total_lost = lost_packets_total_count_[camera_id];
                }
            }
            csv_line_stream << "," << lost_count_since_last << "," << current_total_lost;

            std::lock_guard<std::mutex> csv_lock(csv_queue_mutex_);
            csv_data_queue_.push(csv_line_stream.str());
            csv_queue_cv_.notify_one(); 
            
            // Actualizar estadísticas
            {
                std::lock_guard<std::mutex> stats_lock(metrics_mutex_);
                if (!std::isnan(lat_t0_t1_ms)) all_metrics_[camera_id]["T0_T1_ImgPubToCB_ms"].update(lat_t0_t1_ms);
                if (!std::isnan(lat_t1_t2_ms)) all_metrics_[camera_id]["T1_T2_CBToBatchStart_ms"].update(lat_t1_t2_ms);
                if (!std::isnan(lat_t2_t2a_ms)) all_metrics_[camera_id]["T2_T2a_BatchStartToInfStart_ms"].update(lat_t2_t2a_ms);
                if (!std::isnan(lat_t2a_t2b_ms)) all_metrics_[camera_id]["T2a_T2b_InferenceDuration_ms"].update(lat_t2a_t2b_ms);
                if (!std::isnan(lat_t2b_t3_ms)) all_metrics_[camera_id]["T2b_T3_InfEndToResPub_ms"].update(lat_t2b_t3_ms);
                if (!std::isnan(lat_t3_t4_ms)) all_metrics_[camera_id]["T3_T4_ResPubToRecv_ms"].update(lat_t3_t4_ms);
                if (!std::isnan(lat_segnode_cb_to_segsub_recv_ms)) all_metrics_[camera_id]["Cumulative_T1_T4_ms"].update(lat_segnode_cb_to_segsub_recv_ms);
                if (!std::isnan(lat_total_e2e_ms)) all_metrics_[camera_id]["Cumulative_T0_T4_E2E_ms"].update(lat_total_e2e_ms);

                if (t0_spread_calculated_this_call && !std::isnan(t0_spread_ms)) all_metrics_["batch_global"]["T0_Spread_ms"].update(t0_spread_ms);
                if (t1_spread_calculated_this_call && !std::isnan(t1_spread_ms)) all_metrics_["batch_global"]["T1_Spread_ms"].update(t1_spread_ms);
                if (t2_spread_calculated_this_call && !std::isnan(t2_spread_ms)) all_metrics_["batch_global"]["T2_Spread_ms"].update(t2_spread_ms);
                if (t2a_spread_calculated_this_call && !std::isnan(t2a_spread_ms)) all_metrics_["batch_global"]["T2a_Spread_ms"].update(t2a_spread_ms);
                if (t2b_spread_calculated_this_call && !std::isnan(t2b_spread_ms)) all_metrics_["batch_global"]["T2b_Spread_ms"].update(t2b_spread_ms);
                if (t3_spread_calculated_this_call && !std::isnan(t3_spread_ms)) all_metrics_["batch_global"]["T3_Spread_ms"].update(t3_spread_ms);
                if (t4_spread_calculated_this_call && !std::isnan(t4_spread_ms)) all_metrics_["batch_global"]["T4_Spread_ms"].update(t4_spread_ms);
                
                // T0 offsets (keeping for now)
                if (batch_offsets_calculated_this_call) { 
                    if (!std::isnan(offset_T0_front_vs_left_ms)) {
                        all_metrics_["batch_global"]["OffsetT0_FrontVsLeft_ms"].update(offset_T0_front_vs_left_ms);
                    }
                    if (!std::isnan(offset_T0_right_vs_left_ms)) {
                        all_metrics_["batch_global"]["OffsetT0_RightVsLeft_ms"].update(offset_T0_right_vs_left_ms);
                    }
                }

                // Update global metrics for parallelism if calculated in this call
                if (parallelism_metrics_calculated_this_call) {
                    if (!std::isnan(sum_individual_inf_dur_ms)) all_metrics_["batch_global"]["SumIndividualInfDur_ms"].update(sum_individual_inf_dur_ms);
                    if (!std::isnan(total_batch_inf_span_ms)) all_metrics_["batch_global"]["TotalBatchInfSpan_ms"].update(total_batch_inf_span_ms);
                    if (!std::isnan(inf_overlap_time_ms)) all_metrics_["batch_global"]["InfOverlapTime_ms"].update(inf_overlap_time_ms);
                    if (!std::isnan(parallel_overlap_pct)) all_metrics_["batch_global"]["ParallelOverlap_pct"].update(parallel_overlap_pct);
                    if (!std::isnan(inf_concurrency_factor)) all_metrics_["batch_global"]["InfConcurrencyFactor"].update(inf_concurrency_factor);
                }
            }

            // ... (original frequency counting logic remains the same) ...
            if (camera_id == "left")
            {
                std::lock_guard<std::mutex> lock(msg_count_mutex_left_);
                count_left_++;
            }
            else if (camera_id == "front")
            {
                std::lock_guard<std::mutex> lock(msg_count_mutex_front_);
                count_front_++;
            }
            else if (camera_id == "right")
            {
                std::lock_guard<std::mutex> lock(msg_count_mutex_right_);
                count_right_++;
            }
        };

        // Suscribirse a los tres tópicos referentes a resultados
        rclcpp::QoS qos_profile(rclcpp::KeepLast(5)); 
        qos_profile.reliable();
        qos_profile.durability_volatile();


        sub_left_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            "/segmentation/left/instance_info", qos_profile,
            [callback, this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
            {
                callback(msg, "left");
            });

        sub_front_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            "/segmentation/front/instance_info", qos_profile,
            [callback, this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
            {
                callback(msg, "front");
            });

        sub_right_ = this->create_subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>(
            "/segmentation/right/instance_info", qos_profile,
            [callback, this](const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr msg)
            {
                callback(msg, "right");
            });

        report_timer_ = this->create_wall_timer(
            5s, std::bind(&FrequencySubscriber::report_metrics, this));

        last_report_time_ = this->now();
    }

    ~FrequencySubscriber()
    {   RCLCPP_INFO(this->get_logger(), "Shutting down FrequencySubscriber...");
        stop_csv_writer_thread_ = true;
        csv_queue_cv_.notify_all(); // Notificar al hilo de escritura CSV para que termine
        if (csv_writer_thread_.joinable())
        {
            csv_writer_thread_.join();
            RCLCPP_INFO(this->get_logger(), "CSV writer thread joined successfully.");
        }else
        {
            RCLCPP_WARN(this->get_logger(), "CSV writer thread was not joinable.");
        }
    }

private:

 void initialize_camera_loss_tracking(const std::string& camera_id) {
        // Se llama cuando se recibe el primer paquete de una cámara
        if (first_packet_received_flag_.find(camera_id) == first_packet_received_flag_.end()) {
            std::lock_guard<std::mutex> lock(loss_tracking_mutex_);
            if (first_packet_received_flag_.find(camera_id) == first_packet_received_flag_.end()) { // Doble check por concurrencia
                RCLCPP_INFO(this->get_logger(), "Initializing loss tracking for camera_id: %s", camera_id.c_str());
                last_received_seq_num_[camera_id] = 0; // Se ajustará con el primer paquete real
                lost_packets_total_count_[camera_id] = 0;
                first_packet_received_flag_[camera_id] = false; // Se pondrá a true tras procesar el primer paquete
            }
        }
    }

    void csvWriterLoop()
    {
        std::ofstream latency_log_file_;
        latency_log_file_.open("latency_log.csv", std::ios_base::out | std::ios_base::trunc);

        if(latency_log_file_.is_open())
        {
            latency_log_file_ << "camera_id,packet_seq_num,msg_hdr_seq_nsec," // Añadida columna packet_seq_num
                              << "t0_imgpub_sec,t0_imgpub_nsec,"
                              << "t1_segnode_cb_sec,t1_segnode_cb_nsec,"
                              << "t2_segnode_batchstart_sec,t2_segnode_batchstart_nsec,"
                              // --- New CSV Headers for Inference Timestamps ---
                              << "t2a_inf_start_sec,t2a_inf_start_nsec,"
                              << "t2b_inf_end_sec,t2b_inf_end_nsec,"
                              // --- End New CSV Headers ---
                              << "t3_segnode_respub_sec,t3_segnode_respub_nsec,"
                              << "t4_segsub_resrecv_sec,t4_segsub_resrecv_nsec,"
                              << "lat_imgpub_to_cb_ms,lat_cb_to_batch_start_ms,"
                              // --- New CSV Headers for Latency Breakdown ---
                              << "lat_batchstart_to_infstart_ms," // T2->T2a
                              << "lat_inf_duration_ms,"           // T2a->T2b
                              << "lat_infend_to_respub_ms,"       // T2b->T3
                              // --- End New CSV Headers ---
                              << "lat_res_pub_to_res_recv_ms,"
                              << "lat_segnode_cb_to_segsub_recv_ms,lat_total_e2e_ms,"
                              << "offset_T0_front_vs_left_ms,offset_T0_right_vs_left_ms,"
                              << "t4_reception_spread_ms," // New column for T4 spread
                              << "lost_pkts_since_last,total_lost_pkts_cam\\n"; // Nuevas columnas para pérdida
            RCLCPP_INFO(this->get_logger(), "Logging latency data to latency_log.csv");
        }


        while(true){
            std::string line_to_write;
            {
                std::unique_lock<std::mutex> lock(csv_queue_mutex_);
                csv_queue_cv_.wait(lock, [this] {
                    return stop_csv_writer_thread_ || !csv_data_queue_.empty();
                });

                if (stop_csv_writer_thread_ && csv_data_queue_.empty())
                {
                    break; // Salir del bucle si se indica parar y la cola está vacía
                }

                if (!csv_data_queue_.empty())
                {
                    line_to_write = csv_data_queue_.front();
                    csv_data_queue_.pop();
                }
            } // Se libera el lock

            if (!line_to_write.empty())
            {
                latency_log_file_ << line_to_write << "\n";
            }
        }

        if (latency_log_file_.is_open())
        {
            latency_log_file_.close();
            RCLCPP_INFO(this->get_logger(), "CSV writer thread: Closed latency_log.csv");
        }
    }

    void initialize_colors() {
            class_colors_.push_back(cv::Scalar(255, 0, 0));     // Blue
            class_colors_.push_back(cv::Scalar(0, 255, 0));     // Green
            class_colors_.push_back(cv::Scalar(0, 0, 255));     // Red
            class_colors_.push_back(cv::Scalar(255, 255, 0));   // Yellow
            class_colors_.push_back(cv::Scalar(0, 255, 255));   // Cyan
            class_colors_.push_back(cv::Scalar(255, 0, 255));   // Magenta
            class_colors_.push_back(cv::Scalar(192, 192, 192)); // Silver
            class_colors_.push_back(cv::Scalar(128, 128, 128)); // Gray
            class_colors_.push_back(cv::Scalar(128, 0, 0));     // Maroon
            class_colors_.push_back(cv::Scalar(128, 128, 0));   // Olive
            class_colors_.push_back(cv::Scalar(0, 128, 0));     // Dark Green
            class_colors_.push_back(cv::Scalar(128, 0, 128));   // Purple
            class_colors_.push_back(cv::Scalar(0, 128, 128));   // Teal
            class_colors_.push_back(cv::Scalar(0, 0, 128));     // Navy
            class_colors_.push_back(cv::Scalar(255, 165, 0));   // Orange
            class_colors_.push_back(cv::Scalar(255, 192, 203)); // Pink
            // Añade más si es necesario
        }

    cv::Scalar getRandomTone(const cv::Scalar &base_color, int seed)
    {
        cv::RNG rng(static_cast<uint64_t>(seed)); // Usar uint64_t para el seed de RNG
        double variation_range = 60.0;
        cv::Scalar toned_color;
        for (int i = 0; i < 3; ++i)
        {
            toned_color[i] = cv::saturate_cast<uchar>(base_color[i] + rng.uniform(-variation_range, variation_range));
        }
        return toned_color;
    }

    void save_anomalous_frame_and_log(
        const yolo_custom_interfaces::msg::InstanceSegmentationInfo::SharedPtr& msg,
        const std::string& camera_id,
        const timespec& ts_t3_mono, // Timestamp de publicación del nodo de procesamiento
        const timespec& ts_t4_mono, // Timestamp de recepción en este subscriptor
        double lat_t3_t4_ms)
    {
        // 1. Convertir msg->mask a cv::Mat
        cv_bridge::CvImagePtr cv_ptr_mask;
        try {
            // El encoding de la máscara debe ser el mismo que el publicado por segment_node_3P
            // (ej. "mono16" o "mono8")
            cv_ptr_mask = cv_bridge::toCvCopy(msg->mask, msg->mask.encoding);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "[%s] cv_bridge exception for mask (MsgT1: %u.%09u): %s",
                         camera_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec, e.what());
            return;
        }

        if (!cv_ptr_mask || cv_ptr_mask->image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "[%s] Failed to convert mask to CvImage or mask is empty (MsgT1: %u.%09u).",
                         camera_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
            return;
        }
        const cv::Mat& instance_id_mask = cv_ptr_mask->image; // Debería ser CV_16UC1 o CV_8UC1

        // 2. Crear imagen coloreada
        cv::Mat colored_mask_display = cv::Mat::zeros(instance_id_mask.size(), CV_8UC3);
        size_t num_instances_in_msg = msg->classes.size();

        if (class_colors_.empty()) {
            RCLCPP_INFO(this->get_logger(), "[%s] Color palette is empty. Cannot color anomalous frame.", camera_id.c_str());
            return; // No se puede colorear
        }

        for (int r = 0; r < instance_id_mask.rows; ++r) {
            for (int c = 0; c < instance_id_mask.cols; ++c) {
                uint16_t instance_id_from_mask = 0;
                if (instance_id_mask.type() == CV_16UC1) { // mono16
                    instance_id_from_mask = instance_id_mask.at<uint16_t>(r, c);
                } else if (instance_id_mask.type() == CV_8UC1) { // mono8
                    instance_id_from_mask = static_cast<uint16_t>(instance_id_mask.at<uchar>(r, c));
                } else {
                    if (r == 0 && c == 0) { // Loguear solo una vez por imagen
                         RCLCPP_ERROR_ONCE(this->get_logger(), "[%s] Unexpected mask type: %d for MsgT1: %u.%09u. Cannot color.",
                                          camera_id.c_str(), instance_id_mask.type(), msg->header.stamp.sec, msg->header.stamp.nanosec);
                    }
                   // No se puede procesar, salir de los bucles de coloreado
                   goto end_coloring_loops;
                }

                if (instance_id_from_mask > 0) { // ID 0 es fondo
                    size_t item_idx = static_cast<size_t>(instance_id_from_mask - 1); // IDs son 1-based
                    if (item_idx < num_instances_in_msg && item_idx < msg->classes.size()) {
                        int class_id = msg->classes[item_idx];
                        cv::Scalar base_color = class_colors_[class_id % class_colors_.size()];
                        // Usar el ID de instancia para el seed del color asegura consistencia para la misma instancia
                        cv::Scalar toned_color = getRandomTone(base_color, instance_id_from_mask);
                        colored_mask_display.at<cv::Vec3b>(r, c) = cv::Vec3b(
                            static_cast<uchar>(toned_color[0]),
                            static_cast<uchar>(toned_color[1]),
                            static_cast<uchar>(toned_color[2])
                        );
                    }
                }
            }
        }
        end_coloring_loops:;


        // 3. Guardar imagen
        // Usar el timestamp T1 del header del mensaje para un nombre de archivo único
        anomalous_images_path_ = "/home/david/yolocpp_ws/anomalous_frames/";
        std::string image_filename = anomalous_images_path_ +
                                     camera_id + "_" +
                                     std::to_string(msg->header.stamp.sec) + "_" +
                                     std::to_string(msg->header.stamp.nanosec) + ".png";

        RCLCPP_INFO(this->get_logger(), "[%s] Saving anomalous frame image to %s (MsgT1: %u.%09u)",
                         camera_id.c_str(), image_filename.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
        try {
            if (!colored_mask_display.empty()) {
                cv::imwrite(image_filename, colored_mask_display);
                // RCLCPP_INFO(this->get_logger(), "[%s] Saved anomalous frame image to %s", camera_id.c_str(), image_filename.c_str());
            } else {
                RCLCPP_WARN(this->get_logger(), "[%s] Colored mask display is empty for MsgT1: %u.%09u. Not saving image.",
                            camera_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
            }
        } catch (const cv::Exception& ex) {
            RCLCPP_ERROR(this->get_logger(), "[%s] OpenCV Exception saving image %s (MsgT1: %u.%09u): %s",
                         camera_id.c_str(), image_filename.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec, ex.what());
        }

        // 4. Registrar info adicional en el log CSV de anomalías
        if (anomalous_t3_t4_log_file_.is_open()) {
            std::lock_guard<std::mutex> anomalous_lock(anomalous_log_mutex_);
            anomalous_t3_t4_log_file_ << std::fixed << std::setprecision(3)
                                      << camera_id << ","
                                      << msg->header.stamp.sec << "," << msg->header.stamp.nanosec << "," // T1 del header del mensaje
                                      << ts_t3_mono.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t3_mono.tv_nsec << ","
                                      << ts_t4_mono.tv_sec << "," << std::setfill('0') << std::setw(9) << ts_t4_mono.tv_nsec << ","
                                      << lat_t3_t4_ms << ",\""; // Abrir comillas para listas

            // Clases
            for (size_t i = 0; i < msg->classes.size(); ++i) {
                anomalous_t3_t4_log_file_ << msg->classes[i] << (i == msg->classes.size() - 1 ? "" : ";");
            }
            anomalous_t3_t4_log_file_ << "\",\""; // Cerrar comillas de clases, abrir para scores

            // Scores
            for (size_t i = 0; i < msg->scores.size(); ++i) {
                anomalous_t3_t4_log_file_ << std::fixed << std::setprecision(2) << msg->scores[i] << (i == msg->scores.size() - 1 ? "" : ";");
            }
            anomalous_t3_t4_log_file_ << "\"\n"; // Cerrar comillas de scores y nueva línea
        }
    }
    
    
    void report_metrics() 
    {
        auto now = this->now();
        double elapsed_sec = (now - last_report_time_).seconds();
        if (elapsed_sec <= 0) elapsed_sec = 1.0; 

        uint64_t current_left = 0, current_front = 0, current_right = 0;
        // ... (frequency counting logic remains the same) ...
        {
            std::lock_guard<std::mutex> lock(msg_count_mutex_left_);
            current_left = count_left_;
            count_left_ = 0;
        }
        {
            std::lock_guard<std::mutex> lock(msg_count_mutex_front_);
            current_front = count_front_;
            count_front_ = 0;
        }
        {
            std::lock_guard<std::mutex> lock(msg_count_mutex_right_);
            current_right = count_right_;
            count_right_ = 0;
        }

        double freq_left = current_left / elapsed_sec;
        double freq_front = current_front / elapsed_sec;
        double freq_right = current_right / elapsed_sec;

        RCLCPP_INFO(this->get_logger(),
                    "\n============================= METRICS REPORT (last %.1f sec) ============================", elapsed_sec);
        RCLCPP_INFO(this->get_logger(),
                    "Msg Frequencies -> Left: %.2f Hz, Front: %.2f Hz, Right: %.2f Hz",
                    freq_left, freq_front, freq_right);
        RCLCPP_INFO(this->get_logger(),
                    "--------------------------------------------------------------------------------------");

        std::lock_guard<std::mutex> stats_lock(metrics_mutex_);        
        for (const auto &cam_pair : all_metrics_)
        {
            const std::string &id_key = cam_pair.first; 
            RCLCPP_INFO(this->get_logger(), "Stats for ID: [%s] (Cumulative)", id_key.c_str());
            
            std::vector<std::string> metric_order;
            if (id_key == "batch_global") {
                metric_order = {
                    "T0_Spread_ms", "T1_Spread_ms", "T2_Spread_ms", 
                    "T2a_Spread_ms", "T2b_Spread_ms", "T3_Spread_ms", "T4_Spread_ms",
                    "OffsetT0_FrontVsLeft_ms", "OffsetT0_RightVsLeft_ms"
                };
            } else { 
                metric_order = {
                    "T0_Spread_ms", "T1_Spread_ms", "T2_Spread_ms", 
                    "T2a_Spread_ms", "T2b_Spread_ms", "T3_Spread_ms", "T4_Spread_ms",
                    "SumIndividualInfDur_ms", "TotalBatchInfSpan_ms", "InfOverlapTime_ms", "ParallelOverlap_pct", "InfConcurrencyFactor", // New Parallelism Metrics
                    "OffsetT0_FrontVsLeft_ms", "OffsetT0_RightVsLeft_ms"
                };
            }

            for (const std::string& metric_name : metric_order) {
                auto it = cam_pair.second.find(metric_name);
                if (it != cam_pair.second.end()) {
                    const LatencyMetrics &metrics = it->second;
                    if (metrics.count > 0) {
                        RCLCPP_INFO(this->get_logger(),
                                    "  %-35s: Count=%-5ld, Mean=%-7.3f, Min=%-7.3f, Max=%-7.3f, Var=%-7.3f",
                                    metric_name.c_str(), metrics.count, metrics.mean_ms,
                                    metrics.min_ms, metrics.max_ms, metrics.variance_ms);
                    } else {
                        RCLCPP_INFO(this->get_logger(), "  %-35s: No data yet.", metric_name.c_str());
                    }
                }
            }
            RCLCPP_INFO(this->get_logger(),
                        "--------------------------------------------------------------------------------------");
        }
        RCLCPP_INFO(this->get_logger(),
                    "=========================== END OF METRICS REPORT ======================================");

        last_report_time_ = now;
    }
    

    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_left_;
    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_front_;
    rclcpp::Subscription<yolo_custom_interfaces::msg::InstanceSegmentationInfo>::SharedPtr sub_right_;
    rclcpp::TimerBase::SharedPtr report_timer_;

    rclcpp::Time last_report_time_;

    // Para conteo de frecuencia de mensajes
    std::mutex msg_count_mutex_left_;
    std::mutex msg_count_mutex_front_;
    std::mutex msg_count_mutex_right_;
    uint64_t count_left_ = 0;
    uint64_t count_front_ = 0;
    uint64_t count_right_ = 0;

    // Para log en CSV
    // Variables para el hilo de escritura del CSV
    std::thread csv_writer_thread_;
    std::queue<std::string> csv_data_queue_;
    std::mutex csv_queue_mutex_;
    std::condition_variable csv_queue_cv_;
    std::atomic<bool> stop_csv_writer_thread_;
    std::ofstream anomalous_t3_t4_log_file_; 
    std::mutex anomalous_log_mutex_;
    std::string anomalous_images_path_ = "/home/david/yolocpp_ws/anomalous_frames"; // Ruta para guardar imágenes anómalas
    std::vector<cv::Scalar> class_colors_; // Paleta de colores

    // Para estadísticas de latencia
    std::map<std::string, std::map<std::string, LatencyMetrics>> all_metrics_;
    std::mutex metrics_mutex_;
    std::map<uint64_t, std::map<std::string, timespec>> pending_batch_t0_timestamps_;
    // For T4 reception spread calculation
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t4_timestamps_; // Key: packet_sequence_number
    std::map<uint64_t, double> batch_t4_reception_spread_ms_; // Key: packet_sequence_number
    std::map<uint64_t, std::pair<double, double>> calculated_batch_t0_offsets_; 
    std::mutex batch_data_mutex_; // Kept for T0 offsets, consider merging if T0 spread replaces it

    // For T_X reception spread calculation
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t0_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t1_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t2_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t2a_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t2b_timestamps_;
    std::map<uint64_t, std::map<std::string, timespec>> received_batch_t3_timestamps_;

    std::map<uint64_t, double> batch_t0_spread_ms_;
    std::map<uint64_t, double> batch_t1_spread_ms_;
    std::map<uint64_t, double> batch_t2_spread_ms_;
    std::map<uint64_t, double> batch_t2a_spread_ms_;
    std::map<uint64_t, double> batch_t2b_spread_ms_;
    std::map<uint64_t, double> batch_t3_spread_ms_;
    std::map<uint64_t, double> batch_t4_spread_ms_; 
    std::mutex batch_spread_data_mutex_; 

    // For Parallelism Metrics
    std::map<uint64_t, double> batch_sum_individual_inf_dur_ms_;
    std::map<uint64_t, double> batch_total_inf_span_ms_;
    std::map<uint64_t, double> batch_inf_overlap_time_ms_;
    std::map<uint64_t, double> batch_parallel_overlap_pct_;
    std::map<uint64_t, double> batch_inf_concurrency_factor_;
    std::map<uint64_t, bool> batch_parallelism_metrics_calculated_;

    std::mutex loss_tracking_mutex_;
    std::map<std::string, uint64_t> last_received_seq_num_;      // Key: camera_id, Value: último seq num recibido
    std::map<std::string, uint64_t> lost_packets_total_count_;   // Key: camera_id, Value: total de paquetes perdidos acumulados
    std::map<std::string, bool> first_packet_received_flag_; // Key: camera_id, para manejar el primer paquete
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FrequencySubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}