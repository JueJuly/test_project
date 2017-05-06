
int find_corner(cv::Mat src_img, std::vector<cv::Point2f> &corner_points, 
				bool subpixel_refine = true);
				
		//创建相关模板
		void create_template( float f_angle1, float f_angle2, float f_radius, 
							  std::vector<cv::Mat> &template_vec );
						
				//计算正态概率密度函数
				float compute_normpdf(float x, float mu, float sigma );
			
		//非极大值抑制处理
		void non_max_suppress(	cv::Mat src_mat, int n_region_len, double d_threshold, 
								int n_margin, std::vector<cv::Point2d> &coords_points)
								
		
		#if (/*要亚像素级处理*/)
		//角点坐标亚像素级精简处理
		void corner_coords_subpixel_refine( cv::Mat grad_x_img, cv::Mat grad_y_img, cv::Mat angle_img, 
											cv::Mat weight_img, std::vector<cv::Point2d> corner_coords, 
											std::vector<cv::Point2f> &corner_subpixel_coords, int r );
											
				//寻找梯度方向中两个最大峰值位置
				void edge_orientations( cv::Mat img_angle, cv::Mat img_weight, cv::Mat &v1, cv::Mat &v2 );
						
						//利用meanshift寻找局部最大值
						void find_modes_meanshift( std::vector<float> angle_hist, float sigma, \
													std::vector<float> hist_smoothed, \
													std::vector<ws_Data_f_2d> &modes );
		#endif
											
		//对角点进行评分排序
		void score_corner( cv::Mat src_Mat, cv::Mat angle_img, cv::Mat weight_img, 
							std::vector<cv::Point2d> &coords_points, std::vector<cv::Mat> &template_vec, 
							std::vector<float> &score_corner_table );
							
				//角点相关评分			
				void corner_correlation_score( cv::Mat sub_srcImg, cv::Mat weight_img, 
												std::vector<ws_Data_f_2d> &coords_pts_v1, \
												std::vector<ws_Data_f_2d> &coords_pts_v2 );
						//创建相关模板
						void create_template( float f_angle1, float f_angle2, float f_radius, 
											  std::vector<cv::Mat> &template_vec );