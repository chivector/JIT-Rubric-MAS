# Joint v5 Exact Subset Task Assignments

Numbers are 1-based data-record positions in the pinned source file; CSV header excluded.
Do not renumber after sorting or filtering. JSON maps each row to source_id and stable task_id.
Only selected EVO/VAL/TEST tasks are run; unused is not Dev or a replacement pool.
All v4 source/VAL/TEST boundaries and recorded exposures are preserved.
Manifest SHA256: `e4c2c6735b3625b01335f3942bc1ac256b34ea564eb777950157a347dc4261cd`

## researchrubrics

Revision: `85de3115053d1453ed612caacf4a405edc1ad756`
Data SHA256: `ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb`
Source: https://huggingface.co/datasets/ScaleAI/researchrubrics/tree/85de3115053d1453ed612caacf4a405edc1ad756

### EVOLUTION (20)

6, 13, 17, 20-21, 34-35, 38, 45, 52-53, 60-62, 66, 70, 74, 98, 100-101

### VALIDATION (10)

7, 12, 19, 22, 25-26, 44, 54, 58-59

### TEST (33)

3-5, 10-11, 18, 29-30, 33, 36, 39, 43, 47, 49-50, 55, 65, 69, 71-73, 77-78, 80-81, 83, 85, 87, 91, 93, 95, 97, 99

### UNUSED (38)

1-2, 8-9, 14-16, 23-24, 27-28, 31-32, 37, 40-42, 46, 48, 51, 56-57, 63-64, 67-68, 75-76, 79, 82, 84, 86, 88-90, 92, 94, 96

## deepsearchqa

Revision: `b2623f8653065c2672de6d941fc5434cd652376c`
Data SHA256: `25d48dcf7efa872e5467032e8b8eedf38d301f59a252d0da95cda584baa78396`
Source: https://huggingface.co/datasets/google/deepsearchqa/resolve/b2623f8653065c2672de6d941fc5434cd652376c/DSQA-full.csv

### EVOLUTION (20)

28, 40, 101, 129, 170, 217, 256, 315, 481, 537, 588, 604, 640, 644, 668, 726, 801, 814, 842, 885

### VALIDATION (10)

51, 70, 77, 87, 257, 274, 458, 468, 509, 771

### TEST (50)

6, 26, 62, 67, 91, 93, 105, 118, 127, 133, 137, 148, 198, 216, 237, 239, 262, 265, 289, 299, 313, 347, 352, 387, 412, 420, 477, 480, 482, 525, 541, 561-562, 566, 570, 594, 627, 635, 648, 655, 674, 685, 741, 746, 793, 810, 844, 863, 883, 892

### UNUSED (820)

1-5, 7-25, 27, 29-39, 41-50, 52-61, 63-66, 68-69, 71-76, 78-86, 88-90, 92, 94-100, 102-104, 106-117, 119-126, 128, 130-132, 134-136, 138-147, 149-169, 171-197, 199-215, 218-236, 238, 240-255, 258-261, 263-264, 266-273, 275-288, 290-298, 300-312, 314, 316-346, 348-351, 353-386, 388-411, 413-419, 421-457, 459-467, 469-476, 478-479, 483-508, 510-524, 526-536, 538-540, 542-560, 563-565, 567-569, 571-587, 589-593, 595-603, 605-626, 628-634, 636-639, 641-643, 645-647, 649-654, 656-667, 669-673, 675-684, 686-725, 727-740, 742-745, 747-770, 772-792, 794-800, 802-809, 811-813, 815-841, 843, 845-862, 864-882, 884, 886-891, 893-900

## writingbench

Revision: `ae2d5176449b7b769815482641d35926f26793eb`
Data SHA256: `18fee37c645166eb2e206b36366b2e354265b1e4201db2c86e759e825eaddcbe`
Source: https://raw.githubusercontent.com/X-PLUG/WritingBench/ae2d5176449b7b769815482641d35926f26793eb/benchmark_query/benchmark_all.jsonl

### EVOLUTION (20)

18, 104, 203, 222, 263, 300, 306, 335, 341, 368, 407, 433, 441, 596, 652, 676, 732, 758, 836, 992

### VALIDATION (10)

24, 74, 131, 389, 528, 559, 711, 728, 751, 917

### TEST (50)

12, 21, 41, 47, 57, 75, 78, 89, 95, 115, 129, 133, 152, 161-162, 167, 206, 224, 240, 265, 290, 350, 360, 366, 378, 385, 398, 427, 429, 461, 476, 479, 504, 506, 595, 603, 670, 672, 710, 755, 763, 792, 861, 876, 891, 913, 935, 951, 971, 987

### UNUSED (920)

1-11, 13-17, 19-20, 22-23, 25-40, 42-46, 48-56, 58-73, 76-77, 79-88, 90-94, 96-103, 105-114, 116-128, 130, 132, 134-151, 153-160, 163-166, 168-202, 204-205, 207-221, 223, 225-239, 241-262, 264, 266-289, 291-299, 301-305, 307-334, 336-340, 342-349, 351-359, 361-365, 367, 369-377, 379-384, 386-388, 390-397, 399-406, 408-426, 428, 430-432, 434-440, 442-460, 462-475, 477-478, 480-503, 505, 507-527, 529-558, 560-594, 597-602, 604-651, 653-669, 671, 673-675, 677-709, 712-727, 729-731, 733-750, 752-754, 756-757, 759-762, 764-791, 793-835, 837-860, 862-875, 877-890, 892-912, 914-916, 918-934, 936-950, 952-970, 972-986, 988-991, 993-1000

## deepresearch_bench_ii

Revision: `b38f360603db9531b102aef8c166cedb8509b6f6`
Data SHA256: `263aaabb8c279fb16cbe7c9499afe82d657a8ab3ccfb07ace084387e367d921a`
Source: https://github.com/imlrz/DeepResearch-Bench-II/tree/b38f360603db9531b102aef8c166cedb8509b6f6

### EVOLUTION (0)

none

### VALIDATION (0)

none

### TEST (40)

1, 7, 10, 12, 17-18, 20, 23, 27, 30, 36, 41, 46-48, 55, 59-60, 65, 68, 72-73, 75-76, 81, 84, 86-87, 95-96, 99-100, 107, 112, 116-117, 119, 124-125, 128

### UNUSED (92)

2-6, 8-9, 11, 13-16, 19, 21-22, 24-26, 28-29, 31-35, 37-40, 42-45, 49-54, 56-58, 61-64, 66-67, 69-71, 74, 77-80, 82-83, 85, 88-94, 97-98, 101-106, 108-111, 113-115, 118, 120-123, 126-127, 129-132

### TEST Language en (20)

10, 12, 18, 20, 30, 36, 46, 48, 60, 68, 72, 76, 84, 86, 96, 100, 112, 116, 124, 128

### TEST Language zh (20)

1, 7, 17, 23, 27, 41, 47, 55, 59, 65, 73, 75, 81, 87, 95, 99, 107, 117, 119, 125

## ifeval

Revision: `966cd89545d6b6acfd7638bc708b98261ca58e84`
Data SHA256: `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2`
Source: https://huggingface.co/datasets/google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl

### EVOLUTION (0)

none

### VALIDATION (0)

none

### TEST (50)

37, 47, 57, 59-60, 69, 78, 81, 93, 103, 113, 128, 149, 177, 183, 191, 196, 198, 203, 216-217, 224, 226, 242, 256, 277-278, 291, 309, 311, 335, 343, 354, 358, 380, 382, 400-401, 407, 431-432, 438, 445, 450, 459, 476, 480, 487, 524, 541

### UNUSED (491)

1-36, 38-46, 48-56, 58, 61-68, 70-77, 79-80, 82-92, 94-102, 104-112, 114-127, 129-148, 150-176, 178-182, 184-190, 192-195, 197, 199-202, 204-215, 218-223, 225, 227-241, 243-255, 257-276, 279-290, 292-308, 310, 312-334, 336-342, 344-353, 355-357, 359-379, 381, 383-399, 402-406, 408-430, 433-437, 439-444, 446-449, 451-458, 460-475, 477-479, 481-486, 488-523, 525-540

## ifbench

Revision: `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`
Data SHA256: `d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a`
Source: https://raw.githubusercontent.com/allenai/IFBench/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/data/IFBench_test.jsonl

### EVOLUTION (0)

none

### VALIDATION (0)

none

### TEST (50)

14, 23, 34, 45, 55, 74, 81, 86, 101, 106-108, 110-111, 117-118, 121, 131, 133-134, 138-139, 142, 146, 153, 155, 163-164, 172, 189, 191, 194, 198, 207, 213, 218, 226, 241, 245, 257, 259, 267, 271, 275-276, 286, 289-290, 292, 299

### UNUSED (250)

1-13, 15-22, 24-33, 35-44, 46-54, 56-73, 75-80, 82-85, 87-100, 102-105, 109, 112-116, 119-120, 122-130, 132, 135-137, 140-141, 143-145, 147-152, 154, 156-162, 165-171, 173-188, 190, 192-193, 195-197, 199-206, 208-212, 214-217, 219-225, 227-240, 242-244, 246-256, 258, 260-266, 268-270, 272-274, 277-285, 287-288, 291, 293-298, 300

## Joint Evolution Stages

Each entry is benchmark:source_row.

### Run 0 (seed 20261001)

Stage 1, checkpoint C15:

researchrubrics:60, deepsearchqa:129, writingbench:335, researchrubrics:6, deepsearchqa:170, writingbench:433, researchrubrics:45, deepsearchqa:28, writingbench:732, researchrubrics:70, deepsearchqa:726, writingbench:263, researchrubrics:100, deepsearchqa:481, writingbench:203

Stage 2, checkpoint C30:

researchrubrics:35, deepsearchqa:537, writingbench:306, researchrubrics:61, deepsearchqa:217, writingbench:758, researchrubrics:62, deepsearchqa:315, writingbench:676, researchrubrics:66, deepsearchqa:814, writingbench:992, researchrubrics:21, deepsearchqa:588, writingbench:596

Stage 3, checkpoint C45:

researchrubrics:74, deepsearchqa:885, writingbench:652, researchrubrics:98, deepsearchqa:842, writingbench:407, researchrubrics:53, deepsearchqa:801, writingbench:441, researchrubrics:101, deepsearchqa:40, writingbench:222, researchrubrics:38, deepsearchqa:668, writingbench:836

Stage 4, checkpoint C60:

researchrubrics:20, deepsearchqa:101, writingbench:18, researchrubrics:52, deepsearchqa:604, writingbench:104, researchrubrics:13, deepsearchqa:640, writingbench:341, researchrubrics:17, deepsearchqa:256, writingbench:300, researchrubrics:34, deepsearchqa:644, writingbench:368

### Run 1 (seed 20261002)

Stage 1, checkpoint C15:

researchrubrics:21, deepsearchqa:40, writingbench:104, researchrubrics:53, deepsearchqa:537, writingbench:18, researchrubrics:38, deepsearchqa:481, writingbench:203, researchrubrics:101, deepsearchqa:726, writingbench:596, researchrubrics:61, deepsearchqa:885, writingbench:335

Stage 2, checkpoint C30:

researchrubrics:34, deepsearchqa:640, writingbench:222, researchrubrics:20, deepsearchqa:842, writingbench:341, researchrubrics:45, deepsearchqa:814, writingbench:758, researchrubrics:35, deepsearchqa:28, writingbench:433, researchrubrics:17, deepsearchqa:101, writingbench:306

Stage 3, checkpoint C45:

researchrubrics:52, deepsearchqa:256, writingbench:407, researchrubrics:13, deepsearchqa:129, writingbench:652, researchrubrics:98, deepsearchqa:604, writingbench:263, researchrubrics:60, deepsearchqa:170, writingbench:368, researchrubrics:6, deepsearchqa:644, writingbench:300

Stage 4, checkpoint C60:

researchrubrics:74, deepsearchqa:801, writingbench:992, researchrubrics:62, deepsearchqa:668, writingbench:441, researchrubrics:70, deepsearchqa:315, writingbench:732, researchrubrics:66, deepsearchqa:217, writingbench:676, researchrubrics:100, deepsearchqa:588, writingbench:836

### Run 2 (seed 20261003)

Stage 1, checkpoint C15:

researchrubrics:13, deepsearchqa:101, writingbench:992, researchrubrics:101, deepsearchqa:28, writingbench:652, researchrubrics:74, deepsearchqa:640, writingbench:407, researchrubrics:21, deepsearchqa:315, writingbench:335, researchrubrics:45, deepsearchqa:726, writingbench:263

Stage 2, checkpoint C30:

researchrubrics:38, deepsearchqa:801, writingbench:341, researchrubrics:53, deepsearchqa:588, writingbench:596, researchrubrics:62, deepsearchqa:256, writingbench:758, researchrubrics:60, deepsearchqa:644, writingbench:203, researchrubrics:70, deepsearchqa:170, writingbench:222

Stage 3, checkpoint C45:

researchrubrics:35, deepsearchqa:814, writingbench:306, researchrubrics:61, deepsearchqa:40, writingbench:732, researchrubrics:66, deepsearchqa:217, writingbench:104, researchrubrics:34, deepsearchqa:604, writingbench:368, researchrubrics:6, deepsearchqa:481, writingbench:676

Stage 4, checkpoint C60:

researchrubrics:17, deepsearchqa:537, writingbench:18, researchrubrics:52, deepsearchqa:668, writingbench:836, researchrubrics:98, deepsearchqa:129, writingbench:441, researchrubrics:100, deepsearchqa:842, writingbench:300, researchrubrics:20, deepsearchqa:885, writingbench:433
