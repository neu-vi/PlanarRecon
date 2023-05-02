#Seqential RANSAC and PlaneRecon 3D Evaluation
## Seqential RANSAC
First, run the TSDF-based MVS pipeline, (e.g. Atlas or NeuralRecon), the outputs supposed to be (1) a ```.npz``` file record 
a TSDF volume and the origin and voxel size of the volume, and (2) a ```.ply``` file record the 3D mesh from the TSDF volume. 

Second, change the paths in ```run_seq_RANSAC_mesh.py```, adjust the hyper-params and run it. This code will perform 
sequential RANSAC algorithm on the 3D mesh, and use the TSDF occupancy to check the instance connection. As result, every plane 
instances will be physically unconnected. All vertices in a plane instance will be projected toward the estimated planes to generate 
a piecewise plane reconstruction, named as ```sceneXXXX_XX__ransac_piecewise.ply``` 