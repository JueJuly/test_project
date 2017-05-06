
int find_corner(cv::Mat src_img, std::vector<cv::Point2f> &corner_points, 
				bool subpixel_refine = true);
				
		//�������ģ��
		void create_template( float f_angle1, float f_angle2, float f_radius, 
							  std::vector<cv::Mat> &template_vec );
						
				//������̬�����ܶȺ���
				float compute_normpdf(float x, float mu, float sigma );
			
		//�Ǽ���ֵ���ƴ���
		void non_max_suppress(	cv::Mat src_mat, int n_region_len, double d_threshold, 
								int n_margin, std::vector<cv::Point2d> &coords_points)
								
		
		#if (/*Ҫ�����ؼ�����*/)
		//�ǵ����������ؼ�������
		void corner_coords_subpixel_refine( cv::Mat grad_x_img, cv::Mat grad_y_img, cv::Mat angle_img, 
											cv::Mat weight_img, std::vector<cv::Point2d> corner_coords, 
											std::vector<cv::Point2f> &corner_subpixel_coords, int r );
											
				//Ѱ���ݶȷ�������������ֵλ��
				void edge_orientations( cv::Mat img_angle, cv::Mat img_weight, cv::Mat &v1, cv::Mat &v2 );
						
						//����meanshiftѰ�Ҿֲ����ֵ
						void find_modes_meanshift( std::vector<float> angle_hist, float sigma, \
													std::vector<float> hist_smoothed, \
													std::vector<ws_Data_f_2d> &modes );
		#endif
											
		//�Խǵ������������
		void score_corner( cv::Mat src_Mat, cv::Mat angle_img, cv::Mat weight_img, 
							std::vector<cv::Point2d> &coords_points, std::vector<cv::Mat> &template_vec, 
							std::vector<float> &score_corner_table );
							
				//�ǵ��������			
				void corner_correlation_score( cv::Mat sub_srcImg, cv::Mat weight_img, 
												std::vector<ws_Data_f_2d> &coords_pts_v1, \
												std::vector<ws_Data_f_2d> &coords_pts_v2 );
						//�������ģ��
						void create_template( float f_angle1, float f_angle2, float f_radius, 
											  std::vector<cv::Mat> &template_vec );