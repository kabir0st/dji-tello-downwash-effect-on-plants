# References

## Papers
1. Wang & Schmid, *Action Recognition with Improved Trajectories*, ICCV 2013. [doi:10.1109/ICCV.2013.441](https://doi.org/10.1109/ICCV.2013.441)
   → removing camera motion from optical flow while excluding the foreground (plant mask instead of human boxes)
2. Wang, Oneata, Verbeek & Schmid, *A Robust and Efficient Video Representation for Action Recognition*, IJCV 2016. [doi:10.1007/s11263-015-0846-5](https://doi.org/10.1007/s11263-015-0846-5)
   → journal version of [1]
3. Kroeger, Timofte, Dai & Van Gool, *Fast Optical Flow Using Dense Inverse Search*, ECCV 2016. [doi:10.1007/978-3-319-46493-0_29](https://doi.org/10.1007/978-3-319-46493-0_29)
   → DIS dense optical flow (`dis.calc`)
4. Bergen, Anandan, Hanna & Hingorani, *Hierarchical Model-Based Motion Estimation*, ECCV 1992. [doi:10.1007/3-540-55426-2_27](https://doi.org/10.1007/3-540-55426-2_27)
   → fitting one parametric model to global (camera) motion
5. Odobez & Bouthemy, *Robust Multiresolution Estimation of Parametric Motion Models*, JVCIR 1995. [doi:10.1006/jvci.1995.1029](https://doi.org/10.1006/jvci.1995.1029)
   → robust dominant-motion estimation and subtracting it (`flow - cam`)
6. Irani, Rousso & Peleg, *Computing Occluding and Transparent Motions*, IJCV 1994. [doi:10.1007/BF01420982](https://doi.org/10.1007/BF01420982)
   → separating the dominant background motion from independently moving objects
7. Fischler & Bolles, *Random Sample Consensus*, CACM 1981. [doi:10.1145/358669.358692](https://doi.org/10.1145/358669.358692)
   → RANSAC outlier rejection in the camera fit (e.g. tarp flutter)
8. Carion et al., *SAM 3: Segment Anything with Concepts*, 2025. [arXiv:2511.16719](https://arxiv.org/abs/2511.16719)
   → plant segmentation masks
9. Grundmann, Kwatra & Essa, *Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths*, CVPR 2011. [doi:10.1109/CVPR.2011.5995525](https://doi.org/10.1109/CVPR.2011.5995525)
   → chaining frame-to-frame transforms (`to_first`, `warpAffine`)
10. Gonzalez & Woods, *Digital Image Processing*, 4th ed., 2018, ch. 9
    → erosion for the ~4 px background margin around the plant
11. Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., 2004. [doi:10.1017/CBO9780511811685](https://doi.org/10.1017/CBO9780511811685)
    → pinhole model (apparent size ∝ 1/depth), the basis for `canopy_zoom` (our own adaptation)
12. Vogel, *Drag and Reconfiguration of Broad Leaves in High Winds*, Journal of Experimental Botany 40(8), 1989. [doi:10.1093/jxb/40.8.941](https://doi.org/10.1093/jxb/40.8.941)
    → leaves fold and streamline in strong wind, so extra wind adds less force (README, "Why we think it happens" 3)
13. Leishman, *Principles of Helicopter Aerodynamics*, 2nd ed., Cambridge University Press, 2006, ch. 2
    → momentum theory, induced velocity v = √(T / 2ρA) (README, "Why we think it happens" 1)

## Code
- [OpenCV `cv::DISOpticalFlow`](https://docs.opencv.org/4.x/de/d4f/classcv_1_1DISOpticalFlow.html) → optical flow, MEDIUM preset
- [OpenCV `cv::estimateAffinePartial2D`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) → 4-DOF similarity fit with RANSAC
- OpenCV `cv::erode`, `cv::warpAffine` → background mask margin, warping masks to frame 0
- [LearnOpenCV: Video Stabilization Using Point Feature Matching](https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/) → the `estimateAffinePartial2D` + transform-chaining pattern
- [SAM 3](https://github.com/facebookresearch/sam3) → segmentation model
