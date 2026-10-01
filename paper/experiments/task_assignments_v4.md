# Joint v4 Exact Task Assignments

All numbers below are 1-based data-record positions in the pinned source file (CSV header excluded).
Do not renumber after sorting/filtering. The JSON task_index maps every number to its original source_id and stable task_id.
Membership lists are sorted for reading; execution order is the separately frozen joint_evolution_schedule.
No Dev partition. Former RR quarantine is supplementary TEST, not clean held-out evidence.
Manifest SHA256: `f6a07c37a2cd5505d97461cd923c820767479a94072b4db73c046c0a5e0a0aea`

## researchrubrics

Revision: `85de3115053d1453ed612caacf4a405edc1ad756`
Data SHA256: `ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb`
Source: https://huggingface.co/datasets/ScaleAI/researchrubrics/tree/85de3115053d1453ed612caacf4a405edc1ad756

### EVOLUTION (30)

2, 6, 13, 17, 20-21, 34-35, 38, 41-42, 45, 52-53, 60-62, 64, 66, 68, 70, 74-75, 79, 82, 88-89, 98, 100-101

### VALIDATION (20)

1, 7, 12, 14, 16, 19, 22, 25-27, 40, 44, 54, 56, 58-59, 84, 86, 92, 96

### TEST (51)

3-5, 8-11, 15, 18, 23-24, 28-33, 36-37, 39, 43, 46-51, 55, 57, 63, 65, 67, 69, 71-73, 76-78, 80-81, 83, 85, 87, 90-91, 93-95, 97, 99

### Primary Clean TEST (33)

3-5, 10-11, 18, 29-30, 33, 36, 39, 43, 47, 49-50, 55, 65, 69, 71-73, 77-78, 80-81, 83, 85, 87, 91, 93, 95, 97, 99

### Supplementary Exposed/Reserved TEST (18)

8-9, 15, 23-24, 28, 31-32, 37, 46, 48, 51, 57, 63, 67, 76, 90, 94

## deepsearchqa

Revision: `b2623f8653065c2672de6d941fc5434cd652376c`
Data SHA256: `25d48dcf7efa872e5467032e8b8eedf38d301f59a252d0da95cda584baa78396`
Source: https://huggingface.co/datasets/google/deepsearchqa/resolve/b2623f8653065c2672de6d941fc5434cd652376c/DSQA-full.csv

### EVOLUTION (150)

18, 25, 28, 40, 42, 46, 48, 59, 68, 83, 89, 94, 98, 101, 113, 117, 129, 145, 147, 149, 154, 167, 170, 180, 182-183, 199, 204-205, 207, 212, 217, 221, 226-227, 241, 244, 247-248, 250-251, 253, 256, 273, 280, 285, 291, 305, 309, 314-315, 319, 327, 342, 346, 353, 357, 370, 373, 380, 388, 396, 402, 415, 423, 444, 453, 464, 467, 471, 474, 476, 478, 481, 486, 493-494, 499, 513, 520, 529, 533, 537, 551, 554, 568, 572, 575, 582, 588-590, 592-593, 595, 599, 604, 612, 624, 639-640, 644, 646, 651, 656, 662-664, 668-669, 680, 691, 695, 699, 711-712, 716, 726, 728, 739, 743, 745, 758, 774-775, 783-785, 789, 796, 801, 804, 807, 809, 811, 814, 819, 823, 825, 828, 835-836, 842, 847, 854, 858, 867, 877, 885, 889

### VALIDATION (100)

5, 10, 12, 16-17, 27, 35, 51, 56, 58, 70, 77, 84, 87-88, 100, 103-104, 110, 130, 138, 177-178, 186, 190-191, 206, 208, 222, 234, 242, 249, 257, 268-269, 271, 274-276, 292, 296, 325, 331, 351, 362, 366, 372, 376, 378, 400, 404, 406, 413, 421, 432, 437-438, 442, 445-446, 454, 458, 468, 491, 495-496, 508-509, 512, 516, 531, 536, 614, 621, 629-630, 632, 679, 684, 687, 704, 707-708, 722-723, 748, 764, 768, 771, 777, 779, 799, 815, 833, 845, 848, 850, 868, 875, 900

### TEST (650)

1-4, 6-9, 11, 13-15, 19-24, 26, 29-34, 36-39, 41, 43-45, 47, 49-50, 52-55, 57, 60-67, 69, 71-76, 78-82, 85-86, 90-93, 95-97, 99, 102, 105-109, 111-112, 114-116, 118-128, 131-137, 139-144, 146, 148, 150-153, 155-166, 168-169, 171-176, 179, 181, 184-185, 187-189, 192-198, 200-203, 209-211, 213-216, 218-220, 223-225, 228-233, 235-240, 243, 245-246, 252, 254-255, 258-267, 270, 272, 277-279, 281-284, 286-290, 293-295, 297-304, 306-308, 310-313, 316-318, 320-324, 326, 328-330, 332-341, 343-345, 347-350, 352, 354-356, 358-361, 363-365, 367-369, 371, 374-375, 377, 379, 381-387, 389-395, 397-399, 401, 403, 405, 407-412, 414, 416-420, 422, 424-431, 433-436, 439-441, 443, 447-452, 455-457, 459-463, 465-466, 469-470, 472-473, 475, 477, 479-480, 482-485, 487-490, 492, 497-498, 500-507, 510-511, 514-515, 517-519, 521-528, 530, 532, 534-535, 538-550, 552-553, 555-567, 569-571, 573-574, 576-581, 583-587, 591, 594, 596-598, 600-603, 605-611, 613, 615-620, 622-623, 625-628, 631, 633-638, 641-643, 645, 647-650, 652-655, 657-661, 665-667, 670-678, 681-683, 685-686, 688-690, 692-694, 696-698, 700-703, 705-706, 709-710, 713-715, 717-721, 724-725, 727, 729-738, 740-742, 744, 746-747, 749-757, 759-763, 765-767, 769-770, 772-773, 776, 778, 780-782, 786-788, 790-795, 797-798, 800, 802-803, 805-806, 808, 810, 812-813, 816-818, 820-822, 824, 826-827, 829-832, 834, 837-841, 843-844, 846, 849, 851-853, 855-857, 859-866, 869-874, 876, 878-884, 886-888, 890-899

## writingbench

Revision: `ae2d5176449b7b769815482641d35926f26793eb`
Data SHA256: `18fee37c645166eb2e206b36366b2e354265b1e4201db2c86e759e825eaddcbe`
Source: https://raw.githubusercontent.com/X-PLUG/WritingBench/ae2d5176449b7b769815482641d35926f26793eb/benchmark_query/benchmark_all.jsonl

### EVOLUTION (150)

1, 3, 11, 14-15, 18, 33, 43, 48, 60-61, 65, 67, 70, 76, 80, 104, 141, 145, 151, 153-154, 156, 158, 166, 170-171, 176, 178, 185, 191, 193, 197, 202-203, 205, 222, 229, 249, 255, 260-261, 263, 269, 276, 287, 295, 300, 305-306, 319, 321, 326-327, 334-335, 341, 353, 365, 368, 370-372, 393, 399, 404, 407, 409, 433, 437, 441-442, 449, 480, 493, 498, 501-502, 505, 511, 518-519, 542, 546, 548, 563, 565, 572, 581, 585, 591, 594, 596, 599, 607, 609, 612-613, 618, 622, 628, 631, 638, 644-645, 649, 652, 676, 684, 695, 715, 727, 732, 748, 753, 756-758, 764, 769, 774, 787, 800, 812, 817, 834, 836, 853, 856, 864, 875, 885, 892, 903, 908, 910, 920, 924, 946, 955, 957, 959, 968, 974, 992, 996-1000

### VALIDATION (100)

2, 5, 7-8, 24-25, 58, 74, 81, 83, 105, 121, 131, 159, 163-164, 199, 207, 213, 220, 227, 236, 238-239, 268, 272, 284-286, 294, 296, 310, 324, 331, 336, 343, 348, 358-359, 373, 375, 381-382, 388-389, 411, 424, 432, 452, 458, 473, 477, 525, 528-529, 554, 559, 562, 605, 617, 651, 655, 663, 665, 667, 677, 692, 696, 704, 711-712, 721, 728, 733, 744, 746, 749, 751, 761, 776, 786, 807, 840-841, 849, 863, 870, 874, 888, 899, 917-918, 921-922, 927, 944, 954, 960, 966, 993

### TEST (750)

4, 6, 9-10, 12-13, 16-17, 19-23, 26-32, 34-42, 44-47, 49-57, 59, 62-64, 66, 68-69, 71-73, 75, 77-79, 82, 84-103, 106-120, 122-130, 132-140, 142-144, 146-150, 152, 155, 157, 160-162, 165, 167-169, 172-175, 177, 179-184, 186-190, 192, 194-196, 198, 200-201, 204, 206, 208-212, 214-219, 221, 223-226, 228, 230-235, 237, 240-248, 250-254, 256-259, 262, 264-267, 270-271, 273-275, 277-283, 288-293, 297-299, 301-304, 307-309, 311-318, 320, 322-323, 325, 328-330, 332-333, 337-340, 342, 344-347, 349-352, 354-357, 360-364, 366-367, 369, 374, 376-380, 383-387, 390-392, 394-398, 400-403, 405-406, 408, 410, 412-423, 425-431, 434-436, 438-440, 443-448, 450-451, 453-457, 459-472, 474-476, 478-479, 481-492, 494-497, 499-500, 503-504, 506-510, 512-517, 520-524, 526-527, 530-541, 543-545, 547, 549-553, 555-558, 560-561, 564, 566-571, 573-580, 582-584, 586-590, 592-593, 595, 597-598, 600-604, 606, 608, 610-611, 614-616, 619-621, 623-627, 629-630, 632-637, 639-643, 646-648, 650, 653-654, 656-662, 664, 666, 668-675, 678-683, 685-691, 693-694, 697-703, 705-710, 713-714, 716-720, 722-726, 729-731, 734-743, 745, 747, 750, 752, 754-755, 759-760, 762-763, 765-768, 770-773, 775, 777-785, 788-799, 801-806, 808-811, 813-816, 818-833, 835, 837-839, 842-848, 850-852, 854-855, 857-862, 865-869, 871-873, 876-884, 886-887, 889-891, 893-898, 900-902, 904-907, 909, 911-916, 919, 923, 925-926, 928-943, 945, 947-953, 956, 958, 961-965, 967, 969-973, 975-991, 994-995

## deepresearch_bench_ii

Revision: `b38f360603db9531b102aef8c166cedb8509b6f6`
Data SHA256: `263aaabb8c279fb16cbe7c9499afe82d657a8ab3ccfb07ace084387e367d921a`
Source: https://github.com/imlrz/DeepResearch-Bench-II/tree/b38f360603db9531b102aef8c166cedb8509b6f6

### EVOLUTION (0)

none

### VALIDATION (0)

none

### TEST (132)

1-132

## ifeval

Revision: `966cd89545d6b6acfd7638bc708b98261ca58e84`
Data SHA256: `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2`
Source: https://huggingface.co/datasets/google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl

### EVOLUTION (0)

none

### VALIDATION (0)

none

### TEST (541)

1-541

## ifbench

Revision: `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`
Data SHA256: `d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a`
Source: https://raw.githubusercontent.com/allenai/IFBench/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/data/IFBench_test.jsonl

### EVOLUTION (0)

none

### VALIDATION (0)

none

### TEST (300)

1-300

## Joint Evolution Stages

Each entry is benchmark:source_row. All 3 orders share the same EVO/VAL/TEST membership.

### Run 0 (seed 20261001)

Stage 1, checkpoint C55:

researchrubrics:60, deepsearchqa:423, writingbench:957, deepsearchqa:59, writingbench:591, deepsearchqa:251, writingbench:327, deepsearchqa:346, writingbench:335, deepsearchqa:291, writingbench:185, researchrubrics:6, deepsearchqa:305, writingbench:156, deepsearchqa:453, writingbench:644, deepsearchqa:415, writingbench:817, deepsearchqa:809, writingbench:176, deepsearchqa:25, writingbench:955, researchrubrics:45, deepsearchqa:877, writingbench:607, deepsearchqa:342, writingbench:645, deepsearchqa:147, writingbench:612, deepsearchqa:353, writingbench:892, deepsearchqa:612, writingbench:974, researchrubrics:41, deepsearchqa:662, writingbench:60, deepsearchqa:589, writingbench:70, deepsearchqa:486, writingbench:3, deepsearchqa:167, writingbench:433, deepsearchqa:478, writingbench:732, researchrubrics:70, deepsearchqa:716, writingbench:263, deepsearchqa:89, writingbench:203, deepsearchqa:758, writingbench:61, deepsearchqa:624, writingbench:787, deepsearchqa:646, writingbench:715

Stage 2, checkpoint C110:

researchrubrics:79, deepsearchqa:129, writingbench:306, deepsearchqa:247, writingbench:170, deepsearchqa:785, writingbench:748, deepsearchqa:170, writingbench:493, deepsearchqa:248, writingbench:437, researchrubrics:100, deepsearchqa:28, writingbench:321, deepsearchqa:651, writingbench:305, deepsearchqa:48, writingbench:319, deepsearchqa:728, writingbench:649, deepsearchqa:726, writingbench:353, researchrubrics:35, deepsearchqa:595, writingbench:628, deepsearchqa:513, writingbench:43, deepsearchqa:745, writingbench:1, deepsearchqa:836, writingbench:141, deepsearchqa:796, writingbench:480, researchrubrics:61, deepsearchqa:639, writingbench:276, deepsearchqa:590, writingbench:399, deepsearchqa:18, writingbench:295, deepsearchqa:388, writingbench:758, deepsearchqa:572, writingbench:193, researchrubrics:62, deepsearchqa:280, writingbench:946, deepsearchqa:481, writingbench:546, deepsearchqa:357, writingbench:249, deepsearchqa:712, writingbench:205, deepsearchqa:807, writingbench:875

Stage 3, checkpoint C165:

researchrubrics:88, deepsearchqa:835, writingbench:764, deepsearchqa:227, writingbench:572, deepsearchqa:537, writingbench:885, deepsearchqa:664, writingbench:80, deepsearchqa:467, writingbench:519, researchrubrics:66, deepsearchqa:575, writingbench:853, deepsearchqa:319, writingbench:757, deepsearchqa:554, writingbench:501, deepsearchqa:273, writingbench:151, deepsearchqa:370, writingbench:197, researchrubrics:21, deepsearchqa:823, writingbench:676, deepsearchqa:182, writingbench:409, deepsearchqa:592, writingbench:178, deepsearchqa:83, writingbench:585, deepsearchqa:285, writingbench:756, researchrubrics:2, deepsearchqa:98, writingbench:800, deepsearchqa:493, writingbench:442, deepsearchqa:199, writingbench:518, deepsearchqa:691, writingbench:158, deepsearchqa:327, writingbench:261, researchrubrics:74, deepsearchqa:217, writingbench:11, deepsearchqa:889, writingbench:992, deepsearchqa:315, writingbench:565, deepsearchqa:814, writingbench:903, deepsearchqa:314, writingbench:365

Stage 4, checkpoint C220:

researchrubrics:68, deepsearchqa:783, writingbench:166, deepsearchqa:250, writingbench:769, deepsearchqa:183, writingbench:609, deepsearchqa:858, writingbench:48, deepsearchqa:309, writingbench:67, researchrubrics:98, deepsearchqa:149, writingbench:498, deepsearchqa:789, writingbench:996, deepsearchqa:145, writingbench:631, deepsearchqa:373, writingbench:563, deepsearchqa:669, writingbench:14, researchrubrics:82, deepsearchqa:212, writingbench:999, deepsearchqa:464, writingbench:834, deepsearchqa:520, writingbench:908, deepsearchqa:825, writingbench:65, deepsearchqa:444, writingbench:812, researchrubrics:53, deepsearchqa:588, writingbench:596, deepsearchqa:494, writingbench:727, deepsearchqa:885, writingbench:997, deepsearchqa:811, writingbench:15, deepsearchqa:471, writingbench:864, researchrubrics:89, deepsearchqa:711, writingbench:229, deepsearchqa:68, writingbench:959, deepsearchqa:204, writingbench:652, deepsearchqa:739, writingbench:287, deepsearchqa:474, writingbench:618

Stage 5, checkpoint C275:

researchrubrics:101, deepsearchqa:117, writingbench:638, deepsearchqa:842, writingbench:407, deepsearchqa:801, writingbench:774, deepsearchqa:244, writingbench:613, deepsearchqa:847, writingbench:548, researchrubrics:38, deepsearchqa:402, writingbench:191, deepsearchqa:533, writingbench:260, deepsearchqa:599, writingbench:372, deepsearchqa:40, writingbench:33, deepsearchqa:568, writingbench:441, researchrubrics:20, deepsearchqa:775, writingbench:924, deepsearchqa:154, writingbench:753, deepsearchqa:668, writingbench:920, deepsearchqa:476, writingbench:599, deepsearchqa:241, writingbench:505, researchrubrics:52, deepsearchqa:680, writingbench:371, deepsearchqa:94, writingbench:502, deepsearchqa:774, writingbench:968, deepsearchqa:101, writingbench:326, deepsearchqa:221, writingbench:449, researchrubrics:42, deepsearchqa:42, writingbench:202, deepsearchqa:784, writingbench:153, deepsearchqa:46, writingbench:370, deepsearchqa:867, writingbench:222, deepsearchqa:396, writingbench:404

Stage 6, checkpoint C330:

researchrubrics:13, deepsearchqa:499, writingbench:695, deepsearchqa:828, writingbench:511, deepsearchqa:604, writingbench:581, deepsearchqa:551, writingbench:594, deepsearchqa:593, writingbench:269, researchrubrics:64, deepsearchqa:699, writingbench:622, deepsearchqa:113, writingbench:1000, deepsearchqa:582, writingbench:836, deepsearchqa:253, writingbench:76, deepsearchqa:663, writingbench:154, researchrubrics:75, deepsearchqa:640, writingbench:18, deepsearchqa:256, writingbench:145, deepsearchqa:644, writingbench:393, deepsearchqa:207, writingbench:104, deepsearchqa:226, writingbench:684, researchrubrics:17, deepsearchqa:529, writingbench:171, deepsearchqa:854, writingbench:542, deepsearchqa:180, writingbench:910, deepsearchqa:743, writingbench:856, deepsearchqa:656, writingbench:341, researchrubrics:34, deepsearchqa:819, writingbench:255, deepsearchqa:695, writingbench:998, deepsearchqa:380, writingbench:300, deepsearchqa:205, writingbench:334, deepsearchqa:804, writingbench:368

### Run 1 (seed 20261002)

Stage 1, checkpoint C55:

researchrubrics:21, deepsearchqa:204, writingbench:998, deepsearchqa:247, writingbench:104, deepsearchqa:59, writingbench:287, deepsearchqa:835, writingbench:1, deepsearchqa:784, writingbench:548, researchrubrics:88, deepsearchqa:373, writingbench:18, deepsearchqa:226, writingbench:644, deepsearchqa:529, writingbench:202, deepsearchqa:117, writingbench:885, deepsearchqa:342, writingbench:449, researchrubrics:79, deepsearchqa:493, writingbench:43, deepsearchqa:370, writingbench:631, deepsearchqa:212, writingbench:141, deepsearchqa:669, writingbench:565, deepsearchqa:113, writingbench:834, researchrubrics:53, deepsearchqa:656, writingbench:203, deepsearchqa:183, writingbench:255, deepsearchqa:98, writingbench:67, deepsearchqa:147, writingbench:437, deepsearchqa:18, writingbench:399, researchrubrics:82, deepsearchqa:513, writingbench:596, deepsearchqa:353, writingbench:335, deepsearchqa:575, writingbench:649, deepsearchqa:40, writingbench:295, deepsearchqa:314, writingbench:191

Stage 2, checkpoint C110:

researchrubrics:38, deepsearchqa:537, writingbench:60, deepsearchqa:481, writingbench:222, deepsearchqa:624, writingbench:33, deepsearchqa:464, writingbench:908, deepsearchqa:847, writingbench:542, researchrubrics:101, deepsearchqa:453, writingbench:261, deepsearchqa:471, writingbench:628, deepsearchqa:253, writingbench:269, deepsearchqa:589, writingbench:591, deepsearchqa:241, writingbench:974, researchrubrics:61, deepsearchqa:825, writingbench:957, deepsearchqa:711, writingbench:996, deepsearchqa:807, writingbench:853, deepsearchqa:180, writingbench:11, deepsearchqa:154, writingbench:563, researchrubrics:34, deepsearchqa:828, writingbench:505, deepsearchqa:280, writingbench:61, deepsearchqa:305, writingbench:903, deepsearchqa:823, writingbench:546, deepsearchqa:250, writingbench:518, researchrubrics:20, deepsearchqa:745, writingbench:326, deepsearchqa:699, writingbench:249, deepsearchqa:494, writingbench:480, deepsearchqa:291, writingbench:519, deepsearchqa:205, writingbench:618

Stage 3, checkpoint C165:

researchrubrics:2, deepsearchqa:726, writingbench:341, deepsearchqa:885, writingbench:645, deepsearchqa:476, writingbench:585, deepsearchqa:346, writingbench:920, deepsearchqa:309, writingbench:65, researchrubrics:45, deepsearchqa:582, writingbench:599, deepsearchqa:388, writingbench:493, deepsearchqa:423, writingbench:305, deepsearchqa:89, writingbench:787, deepsearchqa:357, writingbench:817, researchrubrics:35, deepsearchqa:244, writingbench:727, deepsearchqa:94, writingbench:758, deepsearchqa:83, writingbench:684, deepsearchqa:811, writingbench:171, deepsearchqa:474, writingbench:193, researchrubrics:75, deepsearchqa:640, writingbench:753, deepsearchqa:716, writingbench:875, deepsearchqa:380, writingbench:197, deepsearchqa:207, writingbench:433, deepsearchqa:145, writingbench:327, researchrubrics:17, deepsearchqa:554, writingbench:306, deepsearchqa:612, writingbench:748, deepsearchqa:842, writingbench:229, deepsearchqa:593, writingbench:70, deepsearchqa:783, writingbench:774

Stage 4, checkpoint C220:

researchrubrics:89, deepsearchqa:444, writingbench:946, deepsearchqa:774, writingbench:404, deepsearchqa:758, writingbench:572, deepsearchqa:712, writingbench:276, deepsearchqa:273, writingbench:997, researchrubrics:52, deepsearchqa:639, writingbench:968, deepsearchqa:415, writingbench:80, deepsearchqa:814, writingbench:607, deepsearchqa:28, writingbench:764, deepsearchqa:572, writingbench:864, researchrubrics:68, deepsearchqa:285, writingbench:800, deepsearchqa:486, writingbench:158, deepsearchqa:68, writingbench:321, deepsearchqa:101, writingbench:498, deepsearchqa:867, writingbench:812, researchrubrics:13, deepsearchqa:743, writingbench:185, deepsearchqa:595, writingbench:353, deepsearchqa:167, writingbench:372, deepsearchqa:789, writingbench:407, deepsearchqa:819, writingbench:715, researchrubrics:98, deepsearchqa:836, writingbench:924, deepsearchqa:227, writingbench:652, deepsearchqa:256, writingbench:3, deepsearchqa:319, writingbench:48, deepsearchqa:663, writingbench:371

Stage 5, checkpoint C275:

researchrubrics:64, deepsearchqa:129, writingbench:154, deepsearchqa:691, writingbench:955, deepsearchqa:251, writingbench:176, deepsearchqa:533, writingbench:14, deepsearchqa:551, writingbench:638, researchrubrics:60, deepsearchqa:728, writingbench:910, deepsearchqa:199, writingbench:1000, deepsearchqa:604, writingbench:769, deepsearchqa:785, writingbench:581, deepsearchqa:858, writingbench:609, researchrubrics:6, deepsearchqa:877, writingbench:263, deepsearchqa:42, writingbench:442, deepsearchqa:467, writingbench:368, deepsearchqa:590, writingbench:594, deepsearchqa:327, writingbench:856, researchrubrics:74, deepsearchqa:592, writingbench:300, deepsearchqa:170, writingbench:393, deepsearchqa:796, writingbench:151, deepsearchqa:644, writingbench:15, deepsearchqa:396, writingbench:992, researchrubrics:42, deepsearchqa:801, writingbench:145, deepsearchqa:248, writingbench:622, deepsearchqa:668, writingbench:156, deepsearchqa:568, writingbench:441, deepsearchqa:402, writingbench:319

Stage 6, checkpoint C330:

researchrubrics:62, deepsearchqa:646, writingbench:732, deepsearchqa:315, writingbench:613, deepsearchqa:680, writingbench:166, deepsearchqa:478, writingbench:511, deepsearchqa:182, writingbench:999, researchrubrics:41, deepsearchqa:662, writingbench:959, deepsearchqa:804, writingbench:676, deepsearchqa:809, writingbench:695, deepsearchqa:149, writingbench:76, deepsearchqa:499, writingbench:370, researchrubrics:70, deepsearchqa:25, writingbench:170, deepsearchqa:695, writingbench:501, deepsearchqa:217, writingbench:409, deepsearchqa:48, writingbench:205, deepsearchqa:664, writingbench:334, researchrubrics:66, deepsearchqa:854, writingbench:153, deepsearchqa:599, writingbench:612, deepsearchqa:46, writingbench:756, deepsearchqa:588, writingbench:892, deepsearchqa:775, writingbench:365, researchrubrics:100, deepsearchqa:889, writingbench:260, deepsearchqa:221, writingbench:757, deepsearchqa:520, writingbench:836, deepsearchqa:739, writingbench:502, deepsearchqa:651, writingbench:178

### Run 2 (seed 20261003)

Stage 1, checkpoint C55:

researchrubrics:89, deepsearchqa:453, writingbench:154, deepsearchqa:226, writingbench:875, deepsearchqa:309, writingbench:197, deepsearchqa:663, writingbench:409, deepsearchqa:712, writingbench:992, researchrubrics:13, deepsearchqa:590, writingbench:502, deepsearchqa:83, writingbench:638, deepsearchqa:199, writingbench:652, deepsearchqa:373, writingbench:622, deepsearchqa:785, writingbench:287, researchrubrics:101, deepsearchqa:251, writingbench:14, deepsearchqa:807, writingbench:407, deepsearchqa:819, writingbench:968, deepsearchqa:101, writingbench:335, deepsearchqa:529, writingbench:480, researchrubrics:74, deepsearchqa:589, writingbench:263, deepsearchqa:796, writingbench:191, deepsearchqa:789, writingbench:856, deepsearchqa:28, writingbench:326, deepsearchqa:423, writingbench:229, researchrubrics:21, deepsearchqa:716, writingbench:834, deepsearchqa:592, writingbench:853, deepsearchqa:612, writingbench:341, deepsearchqa:486, writingbench:715, deepsearchqa:640, writingbench:498

Stage 2, checkpoint C110:

researchrubrics:45, deepsearchqa:48, writingbench:176, deepsearchqa:145, writingbench:3, deepsearchqa:315, writingbench:892, deepsearchqa:476, writingbench:166, deepsearchqa:825, writingbench:365, researchrubrics:38, deepsearchqa:520, writingbench:596, deepsearchqa:471, writingbench:442, deepsearchqa:811, writingbench:997, deepsearchqa:402, writingbench:769, deepsearchqa:241, writingbench:581, researchrubrics:42, deepsearchqa:624, writingbench:748, deepsearchqa:180, writingbench:645, deepsearchqa:726, writingbench:269, deepsearchqa:575, writingbench:864, deepsearchqa:494, writingbench:955, researchrubrics:82, deepsearchqa:801, writingbench:393, deepsearchqa:588, writingbench:70, deepsearchqa:828, writingbench:764, deepsearchqa:680, writingbench:999, deepsearchqa:572, writingbench:546, researchrubrics:53, deepsearchqa:783, writingbench:957, deepsearchqa:415, writingbench:649, deepsearchqa:280, writingbench:758, deepsearchqa:25, writingbench:618, deepsearchqa:467, writingbench:511

Stage 3, checkpoint C165:

researchrubrics:79, deepsearchqa:327, writingbench:774, deepsearchqa:533, writingbench:80, deepsearchqa:804, writingbench:585, deepsearchqa:774, writingbench:141, deepsearchqa:46, writingbench:203, researchrubrics:75, deepsearchqa:59, writingbench:572, deepsearchqa:478, writingbench:910, deepsearchqa:314, writingbench:43, deepsearchqa:183, writingbench:812, deepsearchqa:835, writingbench:185, researchrubrics:62, deepsearchqa:342, writingbench:908, deepsearchqa:836, writingbench:946, deepsearchqa:221, writingbench:565, deepsearchqa:784, writingbench:631, deepsearchqa:493, writingbench:145, researchrubrics:2, deepsearchqa:568, writingbench:261, deepsearchqa:858, writingbench:170, deepsearchqa:775, writingbench:613, deepsearchqa:669, writingbench:505, deepsearchqa:551, writingbench:548, researchrubrics:41, deepsearchqa:285, writingbench:519, deepsearchqa:305, writingbench:684, deepsearchqa:256, writingbench:974, deepsearchqa:18, writingbench:998, deepsearchqa:651, writingbench:753

Stage 4, checkpoint C220:

researchrubrics:64, deepsearchqa:728, writingbench:924, deepsearchqa:212, writingbench:222, deepsearchqa:98, writingbench:399, deepsearchqa:711, writingbench:1000, deepsearchqa:353, writingbench:60, researchrubrics:60, deepsearchqa:346, writingbench:609, deepsearchqa:244, writingbench:306, deepsearchqa:644, writingbench:171, deepsearchqa:170, writingbench:493, deepsearchqa:814, writingbench:732, researchrubrics:70, deepsearchqa:499, writingbench:563, deepsearchqa:396, writingbench:153, deepsearchqa:889, writingbench:260, deepsearchqa:370, writingbench:449, deepsearchqa:68, writingbench:67, researchrubrics:88, deepsearchqa:291, writingbench:205, deepsearchqa:94, writingbench:305, deepsearchqa:204, writingbench:757, deepsearchqa:40, writingbench:404, deepsearchqa:743, writingbench:319, researchrubrics:35, deepsearchqa:42, writingbench:61, deepsearchqa:691, writingbench:104, deepsearchqa:217, writingbench:334, deepsearchqa:646, writingbench:727, deepsearchqa:695, writingbench:885

Stage 5, checkpoint C275:

researchrubrics:61, deepsearchqa:247, writingbench:996, deepsearchqa:474, writingbench:11, deepsearchqa:582, writingbench:255, deepsearchqa:604, writingbench:591, deepsearchqa:357, writingbench:599, researchrubrics:66, deepsearchqa:481, writingbench:193, deepsearchqa:182, writingbench:607, deepsearchqa:554, writingbench:368, deepsearchqa:513, writingbench:628, deepsearchqa:537, writingbench:959, researchrubrics:34, deepsearchqa:867, writingbench:542, deepsearchqa:668, writingbench:695, deepsearchqa:149, writingbench:353, deepsearchqa:662, writingbench:676, deepsearchqa:593, writingbench:321, researchrubrics:6, deepsearchqa:464, writingbench:800, deepsearchqa:699, writingbench:151, deepsearchqa:656, writingbench:437, deepsearchqa:167, writingbench:76, deepsearchqa:854, writingbench:612, researchrubrics:68, deepsearchqa:739, writingbench:158, deepsearchqa:113, writingbench:18, deepsearchqa:664, writingbench:518, deepsearchqa:599, writingbench:920, deepsearchqa:847, writingbench:370

Stage 6, checkpoint C330:

researchrubrics:17, deepsearchqa:758, writingbench:156, deepsearchqa:823, writingbench:249, deepsearchqa:877, writingbench:756, deepsearchqa:129, writingbench:836, deepsearchqa:842, writingbench:787, researchrubrics:52, deepsearchqa:250, writingbench:178, deepsearchqa:248, writingbench:441, deepsearchqa:207, writingbench:300, deepsearchqa:253, writingbench:433, deepsearchqa:227, writingbench:644, researchrubrics:98, deepsearchqa:639, writingbench:903, deepsearchqa:885, writingbench:33, deepsearchqa:117, writingbench:1, deepsearchqa:380, writingbench:295, deepsearchqa:154, writingbench:501, researchrubrics:100, deepsearchqa:809, writingbench:15, deepsearchqa:205, writingbench:202, deepsearchqa:273, writingbench:371, deepsearchqa:444, writingbench:817, deepsearchqa:147, writingbench:65, researchrubrics:20, deepsearchqa:745, writingbench:594, deepsearchqa:388, writingbench:372, deepsearchqa:595, writingbench:276, deepsearchqa:89, writingbench:48, deepsearchqa:319, writingbench:327
