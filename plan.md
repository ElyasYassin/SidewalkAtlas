Project Name: AI Assistant for Sidewalk Condition Assessment Using Simulation and Geometry-Aware Defect Analysis
Motivation
•	Sidewalk defects pose safety risks (trip hazards) 
•	Manual inspection is: 
	time-consuming 
	subjective 
•	Need an automated, scalable solution 
Goal: Detect, understand, and evaluate sidewalk defects using AI
Key Concept
We combine:
•	Simulation (Issac sim) 
•	AI perception (vision models) 
•	Geometry estimation 
•	Condition scoring 
To move from: Detection → Understanding → Decision
Related Work
1. Sidewalk Defect Detection and Accessibility Analysis
Recent advances in computer vision have enabled automated analysis of sidewalk conditions from image data. Several studies focus on identifying and classifying sidewalk accessibility issues using deep learning models trained on large-scale imagery, such as Google Street View or crowdsourced datasets. For example, prior work has demonstrated the feasibility of classifying sidewalk accessibility conditions—including cracks, uneven surfaces, and obstacles—using deep neural networks and transformer-based architectures [1].
In addition to classification-based approaches, segmentation models have been proposed to delineate defect boundaries at the pixel level. Architectures such as U-Net and Mask R-CNN have been successfully applied to detect and localize pavement defects, achieving high performance in terms of mean Intersection-over-Union (mIoU) and mean Average Precision (mAP) [2]. These approaches provide detailed spatial information about defects, enabling more precise localization compared to classification-only methods.
More recently, research has explored participatory and crowdsourced approaches to sidewalk assessment. For instance, the Sidewalk AI scanner system introduces a web-based platform that enables users to collect sidewalk imagery using smartphones and contribute to large-scale accessibility datasets [3]. The system leverages computer vision algorithms to automatically extract features such as sidewalk width, obstacles, and pavement conditions, enabling scalable and cost-effective urban monitoring. This approach promotes inclusivity and community-driven data collection, offering a novel perspective on urban accessibility analysis.
2. Geometry-Based Defect Measurement
Beyond detection, several studies have explored estimating geometric properties of defects, such as crack length, width, and depth. Traditional methods rely on image processing techniques combined with segmentation outputs to approximate crack dimensions [4]. More advanced systems integrate RGB imagery with depth sensing or LiDAR data to reconstruct 3D representations of pavement defects, enabling more accurate geometric measurements [5].
For example, recent work combines deep learning with 3D point cloud data to quantify defect dimensions and severity levels, improving the reliability of infrastructure assessment [5]. Similarly, crack analysis frameworks extract structural features such as skeletonized crack paths to estimate geometric attributes including length and width [4].
Despite these advancements, existing approaches depend heavily on real-world data acquisition and often require specialized sensors or calibration procedures. Furthermore, geometry estimation is typically treated as an isolated task and is not integrated into broader decision-making frameworks for maintenance planning.
3. Simulation and Synthetic Data in Infrastructure Inspection
Simulation environments have become increasingly important in robotics and computer vision for generating large-scale labeled datasets under controlled conditions. Platforms such as Unity and Unreal Engine enable the creation of realistic virtual environments with precise control over object placement, lighting, and ground-truth annotations [6]. Synthetic data has been widely used to train perception models, particularly when real-world data is scarce or difficult to annotate.
However, in the context of sidewalk inspection, simulation-based approaches remain underexplored. Existing studies rarely incorporate procedural defect modeling or modular defect insertion strategies within simulated environments. Most prior work relies on static image datasets, which lack flexibility and do not allow controlled variation of defect geometry or severity.
4. Limitations of Existing Work
Although significant progress has been made in sidewalk defect detection and accessibility analysis, several limitations remain:
•	Most approaches focus on 2D image-based detection and segmentation without incorporating physical geometry [1][2][3] 
•	Geometry estimation methods often rely on specialized sensors or complex calibration [4][5] 
•	Crowdsourced systems prioritize scalability and accessibility, but lack fine-grained defect analysis and severity assessment [3] 
•	Simulation-based data generation and controlled defect modeling are largely unexplored 
•	Existing systems do not support end-to-end pipelines from perception to maintenance decision-making
5. Research Gap and Motivation
The literature reveals a gap between defect detection and actionable decision-making. While prior work successfully identifies sidewalk defects and accessibility issues, it does not:
•	integrate simulation-based data generation 
•	incorporate geometry-aware defect modeling 
•	provide condition scoring or repair prioritization 
To address these limitations, this study proposes a simulation-driven, geometry-aware sidewalk defect assessment framework, where defect assets are generated and inserted into a virtual environment. The system extends beyond detection by incorporating geometry estimation and condition scoring, ultimately supporting maintenance decision-making and repair prioritization.

Research Overview
Pipeline:
1.	Generate sidewalk environment 
2.	Insert defect assets 
3.	Detect sidewalk region 
4.	Detect and classify defects 
5.	Estimate defect geometry 
6.	Compute condition score 
7.	Generate report on Recommendation repair priority or condition state

Simulation Environment
•	Built in Isaac sim 
•	Uses an existing city-scale simulation environment 
o	e.g., NVIDIA City Asset pack or another available city model 
•	Includes: 
o	sidewalks 
o	roads 
o	curbs 
o	surrounding urban context 
Defect Insertion Strategy
•	Defects are generated as localized assets using Meshy AI 
•	Instead of generating an entire sidewalk scene, we: 
o	identify a small sidewalk segment 
o	replace or modify that local concrete portion 
o	insert the generated defect asset into the existing city model 
Why this strategy?
•	preserves the original city environment 
•	allows the same defect asset to be reused in: 
o	roadside sidewalks 
o	park sidewalks 
o	residential sidewalks 
•	avoids mismatch between generated wide-scene assets and different city models 
Benefit
•	enables controlled, modular, and scalable training data generation 
•	supports realistic defect placement across multiple urban environments

Defect Modeling
We model different defect types:
•	Crack 
•	Uneven slab 
•	Collapsed sidewalk 
•	Missing panel 
Two sources:
•	Meshy → realistic appearance 
•	Synthetic → known geometry 
Sidewalk Detection
•	First step: identify sidewalk region 
•	Methods: 
o	segmentation model 
o	bounding box detection 
Reduces false detection from:
•	road 
•	grass 
•	background 
Defect Detection & Classification
•	Detect defects inside sidewalk region 
•	Classify into categories: 
o	crack 
o	broken 
o	collapsed 
o	uneven 
Standard computer vision task
Geometry Estimation
Estimate physical properties:
•	Crack length 
•	Crack width 
•	Depth / displacement 
•	Damaged area 
Key insight:
Severity depends on geometry, not just type
Condition Scoring
Compute sidewalk condition based on:
•	Defect type 
•	Geometry (size, depth, extent) 
Example:
•	small crack → low risk 
•	large displacement → high risk 
•	collapse → critical 
Repair Decision
Assign repair priority:
•	Low → monitor 
•	Medium → scheduled repair 
•	High → immediate repair 
Supports infrastructure decision-making

Learning Framework
•	Supervised learning for: 
o	sidewalk detection 
o	defect classification 
•	RL learning: 
o	reinforcement learning based decision-making 



Research Task Planning and Feasibility Analysis: 
Simulation:
#	Task	Doable?	Notes
1	Generate sidewalk world (Isaac Sim/ Marble)	Yes	We can import an available asset pack from NVIDIA (e.g., the City Demo Assets Pack). Marble [9] can also be used with Isaac Sim, but it may not be as accurate as the Isaac Lab asset pack in terms of geometry and physics. I have tested the NVIDIA City Demo Asset Pack with Meshy-generated assets, and it works well. However, the limitation is that the city model is not as realistic as Marble, which could be a challenge for real-world deployment. On the other hand, if we try to generate a larger scene with surrounding environments using Meshy from images, it becomes difficult to align assets with different scales, geometry, and textures into a single coherent environment, which can make it messy for training. That’s why I think it would be better to generate specific defect assets and then overlay them onto the city model. This way, we can reuse the same assets across multiple scenes or place them in different locations within the same model.
2	Insert Meshy defect assets (crack, collapse)	Yes	Prepare defect assets for different categories of sidewalk defects using Meshy and overlay them over the Nvidia model’s sidewalk in Isaac Sim.
3	Build simulation scenes for training	Yes	Using Steps 1 and 2, we can generate the simulated world. We can also apply domain randomization (e.g., lighting, textures, and object placement) to increase scene diversity.

Robot Locomotion & Navigation
#	Task	Doable?	Notes
4	Train Spot locomotion on flat ground	Yes	Use built-in Spot locomotion environment training script from Isaac Lab RL resources (e.g., Isaac-Velocity-Flat-Spot-v0). This step teaches the robot how to walk.
5	Deploy trained Spot into sidewalk world	Yes	After locomotion training, spawn the Spot robot into our generated sidewalk environment to test walking stability.
6	Train robot navigation in sidewalk environment	Yes	Train robot to move toward a goal (e.g., stay on sidewalk or reach target). Isaac Sim supports this well.

Perception
#	Task	Doable?	Notes
7	Train robot to detect sidewalk	Yes	Use a segmentation model to identify the sidewalk region, or a detection model if needed. Models such as SegFormer or DeepLabV3 can be used for accurate sidewalk region understanding.
8	Detect defects inside sidewalk region	Yes	Detect defects within the segmented sidewalk region using object detection (e.g., DETR, YOLOv8, or HuggingFace models). 
9	Classify defect type (crack, broken, etc.)	Yes	Classify defects into different sidewalk defect categories using standard computer vision approaches. We can apply transfer learning with modern models such as EfficientNet or other updated architectures. If a labeled dataset is available, those images can be used both for model training and for developing corresponding defect assets, using the same labels as ground truth. For now, we can start with around 50 well-labeled images across different categories, generate corresponding defect scenes in Isaac Sim, and then expand the dataset by creating multiple variations through domain randomization.



Geometry & Measurement
#	Task	Doable?	Notes
10	Estimate defect geometry (length, width, depth)	Needs more careful thought	Possible, but we need to verify accuracy. 2D images alone are not reliable; depth sensing or synthetic labeling may be needed.
11	Use Meshy assets as ground truth for geometry	Needs more careful thought	Meshy ≠ physically accurate measurements, so not reliable as ground truth.
12	Generate synthetic defects with known geometry	Yes (Need more study)	Strong approach for RL and scoring. Generate controlled shapes with known dimensions.
13	Combine real-like (Meshy) + synthetic geometry data	Yes (Need more study)	Hybrid approach is best: Meshy for realism, synthetic for accurate measurements.

Condition Assessment & Decision Making
#	Task	Doable?	Notes
14	Compute condition score from geometry	Yes (Need more study)	Can be rule-based or ML-based.
15	Decide repair urgency (immediate/later) or sidewalk condition (poor, excellent)	Yes	Use Denver sidewalk rules (measurement thresholds) and dataset for categorization.
16	Train RL model for scoring/decision	Optional	Not required initially. Can be added later for automated decision optimization.

Real-World Deployment
#	Task	Doable?	Notes
17	Transfer model to real-world data with Spot robot	Needs care	Requires domain adaptation (simulation-to-real gap handling, lighting, noise, sensor differences).


















References:
[1] Saha, S., et al. Sidewalk Accessibility Classification Using Deep Learning, ACM ASSETS, 2024.
[2] Zhang, L., et al. Deep Learning-Based Pavement Crack Detection Using U-Net, Automation in Construction, 2018.
[3] Sidewalk AI Scanner: Crowdsourced Sidewalk Accessibility Mapping, Royal Society, 2023.
[4] Dorafshan, S., et al. Automated Crack Detection and Measurement, Computer-Aided Civil and Infrastructure Engineering, 2018.
[5] Chen, F., et al. 3D Pavement Defect Detection Using LiDAR and Deep Learning, IEEE T-ITS, 2022.
[6] Dosovitskiy, A., et al. CARLA: An Open Urban Driving Simulator, CoRL, 2017.
[7] Nvidia Omniverse USD. Source : https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html#city-demo-assets-pack
[8] Introducing Supervisely Synthetic Crack Segmentation Dataset. Source: https://supervisely.com/blog/introducing-supervisely-synthetic-crack-segmentation-dataset/
[9] Simulate Robotic Environments Faster with NVIDIA Isaac Sim and World Labs Marble, link: https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble/
