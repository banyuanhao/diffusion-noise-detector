### **Question 5: Limited performance of the detector**

## The main reasons for the limited performance and their remedies are as follows:

### 1. Drawbacks in Ground Truth Labeling:
As shown in Fig. 4 and Fig. 5, random sampled noise can contain multiple or no trigger patches, which was not considered when generating ground truth labels. Specifically:  
- **No trigger patch:** When noise contains no trigger patch, it results in unreasonable data points.  
- **Multiple trigger patches:** Bounding boxes may concentrate on one patch while ignoring others, leading to unbalanced data points.  
- **Noisy labels:** Each noise can generate up to 25 bounding boxes, many of which are noisy or repetitive.  

**Remedy:** Analyze the statistics of bounding boxes for each noise to extract trigger patch positions. By clustering bounding boxes (similar to Fig. 5), we can compute Within-Cluster Variance (WCV) and Between-Cluster Variance (BCV). If WCV exceeds a threshold, the noise lacks a trigger patch and should be filtered out. Each cluster can represent a trigger patch, and the averaged bounding box for each cluster can serve as the ground truth label.

---

### 2. Challenges with Advanced Techniques:
Advanced techniques commonly used in object detection, like random cropping in data augmentation, reduced performance for our task due to the unique nature of our input data. Simplified configurations were more effective, retaining statistical properties. Ultimately, we adopted only random flip and crop for data augmentation.

---

### 3. Object Size Variance:
Object sizes in our dataset vary significantly. For instance, in a 512x512 image, the average area of a bear is 69,274 pixels, whereas for a sports ball, it is only 29,919 pixels. Real-world scenarios also involve significant object size variation.  

**Remedy:** Future training should include more objects of varied sizes to improve detector robustness in real-world applications.
