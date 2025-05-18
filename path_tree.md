```
MultiCoil
|-- BlackBlood
|   `-- TrainingSet
|       |-- FullSample
|       |   `-- Center006
|       |       `-- Siemens_30T_Prisma
|       |           |-- P002
|       |           |   |-- blackblood.mat
|       |           |   `-- blackblood_info.csv
|       |           |-- P005
|       |           |   |-- blackblood.mat
|       |           |   `-- blackblood_info.csv
|       |           `-- ... (13 more items)
|       |-- ImageShow
|       |   `-- Center006
|       |       `-- Siemens_30T_Prisma
|       |           |-- P002
|       |           |   |-- magnitude
|       |           |   |   |-- blackblood_slice10_time1_mag.png
|       |           |   |   |-- blackblood_slice1_time1_mag.png
|       |           |   |   `-- ... (8 more items)
|       |           |   |-- nii
|       |           |   |   |-- magnitude
|       |           |   |   |   |-- blackblood_slice10_time1_mag.nii
|       |           |   |   |   |-- blackblood_slice1_time1_mag.nii
|       |           |   |   |   `-- ... (8 more items)
|       |           |   |   `-- phase
|       |           |   |       |-- blackblood_slice10_time1_phase.nii
|       |           |   |       |-- blackblood_slice1_time1_phase.nii
|       |           |   |       `-- ... (8 more items)
|       |           |   `-- phase
|       |           |       |-- blackblood_slice10_time1_phase.png
|       |           |       |-- blackblood_slice1_time1_phase.png
|       |           |       `-- ... (8 more items)
|       |           |-- P005
|       |           |   |-- magnitude
|       |           |   |   |-- blackblood_slice1_time1_mag.png
|       |           |   |   |-- blackblood_slice2_time1_mag.png
|       |           |   |   `-- ... (7 more items)
|       |           |   |-- nii
|       |           |   |   |-- magnitude
|       |           |   |   |   |-- blackblood_slice1_time1_mag.nii
|       |           |   |   |   |-- blackblood_slice2_time1_mag.nii
|       |           |   |   |   `-- ... (7 more items)
|       |           |   |   `-- phase
|       |           |   |       |-- blackblood_slice1_time1_phase.nii
|       |           |   |       |-- blackblood_slice2_time1_phase.nii
|       |           |   |       `-- ... (7 more items)
|       |           |   `-- phase
|       |           |       |-- blackblood_slice1_time1_phase.png
|       |           |       |-- blackblood_slice2_time1_phase.png
|       |           |       `-- ... (7 more items)
|       |           `-- ... (12 more items)
|       `-- Mask_TaskAll
|           `-- Center006
|               `-- Siemens_30T_Prisma
|                   |-- P002
|                   |   |-- blackblood_mask_Uniform16.mat
|                   |   |-- blackblood_mask_Uniform24.mat
|                   |   `-- ... (7 more items)
|                   |-- P005
|                   |   |-- blackblood_mask_Uniform16.mat
|                   |   |-- blackblood_mask_Uniform24.mat
|                   |   `-- ... (7 more items)
|                   `-- ... (13 more items)
|-- Cine
|   `-- TrainingSet
|       |-- FullSample
|       |   |-- Center001
|       |   |   `-- UIH_30T_umr780
|       |   |       |-- P001
|       |   |       |   |-- cine_lax_3ch.mat
|       |   |       |   |-- cine_lax_3ch_info.csv
|       |   |       |   |-- cine_lax_4ch.mat
|       |   |       |   `-- cine_lax_4ch_info.csv
|       |   |       |-- P002
|       |   |       |   |-- cine_sax.mat
|       |   |       |   `-- cine_sax_info.csv
|       |   |       `-- ... (46 more items)
|       |   |-- Center002
|       |   |   |-- Siemens_30T_CIMA.X
|       |   |   |   |-- P001
|       |   |   |   |   |-- cine_lax.mat
|       |   |   |   |   `-- cine_lax_info.csv
|       |   |   |   |-- P002
|       |   |   |   |   |-- cine_lax.mat
|       |   |   |   |   `-- cine_lax_info.csv
|       |   |   |   `-- ... (13 more items)
|       |   |   `-- UIH_30T_umr880
|       |   |       |-- P001
|       |   |       |   |-- cine_lax_2ch.mat
|       |   |       |   |-- cine_lax_2ch_info.csv
|       |   |       |   `-- ... (8 more items)
|       |   |       |-- P002
|       |   |       |   |-- cine_lax_2ch.mat
|       |   |       |   |-- cine_lax_2ch_info.csv
|       |   |       |   `-- ... (6 more items)
|       |   |       `-- ... (7 more items)
|       |   `-- ... (4 more items)
|       |-- ImageShow
|       |   |-- Center001
|       |   |   `-- UIH_30T_umr780
|       |   |       |-- P001
|       |   |       |   |-- magnitude
|       |   |       |   |   |-- cine_lax_3ch_slice1_time10_mag.png
|       |   |       |   |   |-- cine_lax_3ch_slice1_time11_mag.png
|       |   |       |   |   `-- ... (22 more items)
|       |   |       |   |-- nii
|       |   |       |   |   |-- magnitude
|       |   |       |   |   |   |-- cine_lax_3ch_slice1_time10_mag.nii
|       |   |       |   |   |   |-- cine_lax_3ch_slice1_time11_mag.nii
|       |   |       |   |   |   `-- ... (22 more items)
|       |   |       |   |   `-- phase
|       |   |       |   |       |-- cine_lax_3ch_slice1_time10_phase.nii
|       |   |       |   |       |-- cine_lax_3ch_slice1_time11_phase.nii
|       |   |       |   |       `-- ... (22 more items)
|       |   |       |   `-- phase
|       |   |       |       |-- cine_lax_3ch_slice1_time10_phase.png
|       |   |       |       |-- cine_lax_3ch_slice1_time11_phase.png
|       |   |       |       `-- ... (22 more items)
|       |   |       |-- P002
|       |   |       |   |-- magnitude
|       |   |       |   |   |-- cine_sax_slice1_time10_mag.png
|       |   |       |   |   |-- cine_sax_slice1_time11_mag.png
|       |   |       |   |   `-- ... (22 more items)
|       |   |       |   |-- nii
|       |   |       |   |   |-- magnitude
|       |   |       |   |   |   |-- cine_sax_slice1_time10_mag.nii
|       |   |       |   |   |   |-- cine_sax_slice1_time11_mag.nii
|       |   |       |   |   |   `-- ... (22 more items)
|       |   |       |   |   `-- phase
|       |   |       |   |       |-- cine_sax_slice1_time10_phase.nii
|       |   |       |   |       |-- cine_sax_slice1_time11_phase.nii
|       |   |       |   |       `-- ... (22 more items)
|       |   |       |   `-- phase
|       |   |       |       |-- cine_sax_slice1_time10_phase.png
|       |   |       |       |-- cine_sax_slice1_time11_phase.png
|       |   |       |       `-- ... (22 more items)
|       |   |       `-- ... (46 more items)
|       |   |-- Center002
|       |   |   |-- Siemens_30T_CIMA.X
|       |   |   |   |-- P001
|       |   |   |   |   |-- magnitude
|       |   |   |   |   |   |-- cine_lax_slice1_time10_mag.png
|       |   |   |   |   |   |-- cine_lax_slice1_time11_mag.png
|       |   |   |   |   |   `-- ... (34 more items)
|       |   |   |   |   |-- nii
|       |   |   |   |   |   |-- magnitude
|       |   |   |   |   |   |   |-- cine_lax_slice1_time10_mag.nii
|       |   |   |   |   |   |   |-- cine_lax_slice1_time11_mag.nii
|       |   |   |   |   |   |   `-- ... (34 more items)
|       |   |   |   |   |   `-- phase
|       |   |   |   |   |       |-- cine_lax_slice1_time10_phase.nii
|       |   |   |   |   |       |-- cine_lax_slice1_time11_phase.nii
|       |   |   |   |   |       `-- ... (34 more items)
|       |   |   |   |   `-- phase
|       |   |   |   |       |-- cine_lax_slice1_time10_phase.png
|       |   |   |   |       |-- cine_lax_slice1_time11_phase.png
|       |   |   |   |       `-- ... (34 more items)
|       |   |   |   |-- P002
|       |   |   |   |   |-- magnitude
|       |   |   |   |   |   |-- cine_lax_slice1_time10_mag.png
|       |   |   |   |   |   |-- cine_lax_slice1_time11_mag.png
|       |   |   |   |   |   `-- ... (34 more items)
|       |   |   |   |   |-- nii
|       |   |   |   |   |   |-- magnitude
|       |   |   |   |   |   |   |-- cine_lax_slice1_time10_mag.nii
|       |   |   |   |   |   |   |-- cine_lax_slice1_time11_mag.nii
|       |   |   |   |   |   |   `-- ... (34 more items)
|       |   |   |   |   |   `-- phase
|       |   |   |   |   |       |-- cine_lax_slice1_time10_phase.nii
|       |   |   |   |   |       |-- cine_lax_slice1_time11_phase.nii
|       |   |   |   |   |       `-- ... (34 more items)
|       |   |   |   |   `-- phase
|       |   |   |   |       |-- cine_lax_slice1_time10_phase.png
|       |   |   |   |       |-- cine_lax_slice1_time11_phase.png
|       |   |   |   |       `-- ... (34 more items)
|       |   |   |   `-- ... (13 more items)
|       |   |   `-- UIH_30T_umr880
|       |   |       |-- P001
|       |   |       |   |-- magnitude
|       |   |       |   |   |-- cine_lax_2ch_slice1_time10_mag.png
|       |   |       |   |   |-- cine_lax_2ch_slice1_time11_mag.png
|       |   |       |   |   `-- ... (130 more items)
|       |   |       |   |-- nii
|       |   |       |   |   |-- magnitude
|       |   |       |   |   |   |-- cine_lax_2ch_slice1_time10_mag.nii
|       |   |       |   |   |   |-- cine_lax_2ch_slice1_time11_mag.nii
|       |   |       |   |   |   `-- ... (130 more items)
|       |   |       |   |   `-- phase
|       |   |       |   |       |-- cine_lax_2ch_slice1_time10_phase.nii
|       |   |       |   |       |-- cine_lax_2ch_slice1_time11_phase.nii
|       |   |       |   |       `-- ... (130 more items)
|       |   |       |   `-- phase
|       |   |       |       |-- cine_lax_2ch_slice1_time10_phase.png
|       |   |       |       |-- cine_lax_2ch_slice1_time11_phase.png
|       |   |       |       `-- ... (130 more items)
|       |   |       |-- P002
|       |   |       |   |-- magnitude
|       |   |       |   |   |-- cine_lax_2ch_slice1_time10_mag.png
|       |   |       |   |   |-- cine_lax_2ch_slice1_time11_mag.png
|       |   |       |   |   `-- ... (118 more items)
|       |   |       |   |-- nii
|       |   |       |   |   |-- magnitude
|       |   |       |   |   |   |-- cine_lax_2ch_slice1_time10_mag.nii
|       |   |       |   |   |   |-- cine_lax_2ch_slice1_time11_mag.nii
|       |   |       |   |   |   `-- ... (118 more items)
|       |   |       |   |   `-- phase
|       |   |       |   |       |-- cine_lax_2ch_slice1_time10_phase.nii
|       |   |       |   |       |-- cine_lax_2ch_slice1_time11_phase.nii
|       |   |       |   |       `-- ... (118 more items)
|       |   |       |   `-- phase
|       |   |       |       |-- cine_lax_2ch_slice1_time10_phase.png
|       |   |       |       |-- cine_lax_2ch_slice1_time11_phase.png
|       |   |       |       `-- ... (118 more items)
|       |   |       `-- ... (7 more items)
|       |   `-- ... (4 more items)
|       `-- Mask_TaskAll
|           |-- Center001
|           |   `-- UIH_30T_umr780
|           |       |-- P001
|           |       |   |-- cine_lax_3ch_mask_Uniform16.mat
|           |       |   |-- cine_lax_3ch_mask_Uniform24.mat
|           |       |   `-- ... (16 more items)
|           |       |-- P002
|           |       |   |-- cine_sax_mask_Uniform16.mat
|           |       |   |-- cine_sax_mask_Uniform24.mat
|           |       |   `-- ... (7 more items)
|           |       `-- ... (46 more items)
|           |-- Center002
|           |   |-- Siemens_30T_CIMA.X
|           |   |   |-- P001
|           |   |   |   |-- cine_lax_mask_Uniform16.mat
|           |   |   |   |-- cine_lax_mask_Uniform24.mat
|           |   |   |   `-- ... (7 more items)
|           |   |   |-- P002
|           |   |   |   |-- cine_lax_mask_Uniform16.mat
|           |   |   |   |-- cine_lax_mask_Uniform24.mat
|           |   |   |   `-- ... (7 more items)
|           |   |   `-- ... (13 more items)
|           |   `-- UIH_30T_umr880
|           |       |-- P001
|           |       |   |-- cine_lax_2ch_mask_Uniform16.mat
|           |       |   |-- cine_lax_2ch_mask_Uniform24.mat
|           |       |   `-- ... (43 more items)
|           |       |-- P002
|           |       |   |-- cine_lax_2ch_mask_Uniform16.mat
|           |       |   |-- cine_lax_2ch_mask_Uniform24.mat
|           |       |   `-- ... (34 more items)
|           |       `-- ... (7 more items)
|           `-- ... (4 more items)
`-- ... (9 more items)
```
