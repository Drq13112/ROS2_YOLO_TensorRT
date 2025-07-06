
# [PointCloud2 message explained](https://medium.com/@tonyjacob_/pointcloud2-message-explained-853bd9907743)

PointCloud2 is the successor of PointCloud, and it is supposed to be more efficient.

## Message Structure (sensor_msgs/PointCloud2)
### Message Definition
This message holds a collection of N-dimensional points, which may contain additional information such as normals, intensity, etc.
The point data is stored as a binary blob, its layout described by the contents of the "fields" array.

The point cloud data may be organized in 2D (image-like) or 1D (unordered). Point clouds organized as 2D images may be produced by depth sensors such as stereo or time-of-flight.

Time of sensor data acquisition, and the coordinate frame ID (for 3D points)
`std_msgs/Header header`

2D structure of the point cloud. If the cloud is unordered, height is 1 and width is the length of the point cloud.
`uint32 height # Denotes the number of vertical channels the device has. e.g. 128`
`uint32 width # Denotes the number of points per vertical channel`

Describes the channels and their layout in the binary data blob
`sensor_msgs/PointField[] fields`
This is like a C++ Struct. Each field, corresponds to a particular attribute of th point cloud you have. It takes a name (attribute), offset (start index of the point), datatype and count as its argument. The data is a is arranged in 1D in groups of fields, e.g. [x1, y1, x2, y2, ..., xn, yn]

`bool is_bigendian # Is this data bigendian direction by which the bytes are read`
`uint32 point_step # Length of a point in bytes = length(fields) * dtype`
`uint32 row_step # Length of a row in bytes = height * width * point_step`
`uint8[] data # Actual point data, size is (row_step * height)`

`bool is_dense # True if there are no invalid points`

### C++ Boilerplate Code 

    #include <sensor_msgs/PointCloud2.h>  
    #include <sensor_msgs/PointField.h>  
    #include <sensor_msgs/point_cloud2_iterator.h>  
      
      
    sensor_msgs::PointCloud2 pcl_msg;  
      
    //Modifier to describe what the fields are.  
    sensor_msgs::PointCloud2Modifier modifier(pcl_msg);  
      
    modifier.setPointCloud2Fields(4,  
    "x", 1, sensor_msgs::PointField::FLOAT32,  
    "y", 1, sensor_msgs::PointField::FLOAT32,  
    "z", 1, sensor_msgs::PointField::FLOAT32,  
    "intensity", 1, sensor_msgs::PointField::FLOAT32);  
      
    //Msg header  
    pcl_msg.header = std_msgs::Header();  
    pcl_msg.header.stamp = ros::Time::now();  
    pcl_msg.header.frame_id = "frame";  
      
    pcl_msg.height = height;  
    pcl_msg.width = width;  
    pcl_msg.is_dense = true;  
      
    //Total number of bytes per point  
    pcl_msg.point_step = 16;  
    pcl_msg.row_step = pcl_msg.point_step * pcl_msg.width * pcl_msg.height;  
    pcl_msg.data.resize(pcl_msg.row_step);  
      
    //Iterators for PointCloud msg  
    sensor_msgs::PointCloud2Iterator<float> iterX(pcl_msg, "x");  
    sensor_msgs::PointCloud2Iterator<float> iterY(pcl_msg, "y");  
    sensor_msgs::PointCloud2Iterator<float> iterZ(pcl_msg, "z");  
    sensor_msgs::PointCloud2Iterator<float> iterIntensity(pcl_msg, "intensity");  
      
    //iterate over the message and populate the fields.  
    {  
    *iterX = //Your x data  
    *iterY = //Your y data  
    *iterZ = //Your z data  
    *iterIntensity = //Your intensity data  
      
    // Increment the iterators  
    ++iterX;  
    ++iterY;  
    ++iterZ;  
    ++iterIntensity;  
    }


