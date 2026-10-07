# Independent v6 Domain-Balanced Task Assignments

This is a new registration; the historical v5/v6 manifests and campaigns remain unchanged.
Numbers are 1-based data-record positions in the pinned source file; CSV header excluded.
Each subset balances available primary domains. Rare or unavailable domains are reported below.
ResearchRubrics retains 30 parent EVO and 33 historical clean-designated TEST tasks; it balances ten VAL choices within the 20 parent VAL tasks.
WritingBench balances six primary domains, then en/zh inside each domain and over the subset.
DeepResearch Bench II balances themes separately within 20 en and 20 zh tasks.
Five registered target exposures are excluded from VAL/TEST. Fixed historical RR TEST retains a marked supplemental exposure; unknown historical exposure remains an audit limitation.
Manifest SHA256: `4cd28cf7008595c7194ee4979de20cced03d8b58a41bcee4de1a2c50e5cf7630`

## researchrubrics

Revision: `85de3115053d1453ed612caacf4a405edc1ad756`
Data SHA256: `ea2023d03953b70ba4a2f7f1fe490b434f18cec26c4fdb48acc0e3edf3d8edfb`
Source: https://huggingface.co/datasets/ScaleAI/researchrubrics/tree/85de3115053d1453ed612caacf4a405edc1ad756

### Domain Distribution

| Domain | EVO | VAL | TEST |
| --- | ---: | ---: | ---: |
| AI & ML | 5 | 1 | 4 |
| Business Planning & Research | 5 | 1 | 4 |
| Creative Writing | 3 | 1 | 1 |
| Current Events | 3 | 1 | 1 |
| General Consumer Research | 5 | 1 | 5 |
| Historical Analysis | 6 | 1 | 4 |
| Hypotheticals & Philosophy | 3 | 0 | 5 |
| Other | 3 | 2 | 1 |
| STEM | 2 | 1 | 3 |
| Technical Documentation | 5 | 1 | 5 |

### EVOLUTION (40)

1-2, 6, 12-14, 17, 20-21, 34-35, 38, 40-42, 45, 52-54, 59-62, 64, 66, 68, 70, 74-75, 79, 82, 84, 86, 88-89, 92, 96, 98, 100-101

Policy: retain 30 original parent EVO tasks and transfer the 10 parent VAL tasks not selected for VAL; no EVO-to-VAL transfer

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 4.000 tasks/domain; absolute deviation: 12.000

### VALIDATION (10)

7, 16, 19, 22, 25-27, 44, 56, 58

Policy: equal domain quotas within the original parent partition

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 1.000 tasks/domain; absolute deviation: 2.000

Absent from eligible parent pool: Hypotheticals & Philosophy

### TEST (33)

3-5, 10-11, 18, 29-30, 33, 36, 39, 43, 47, 49-50, 55, 65, 69, 71-73, 77-78, 80-81, 83, 85, 87, 91, 93, 95, 97, 99

Policy: retain all 33 historically designated clean-unseen parent TEST tasks

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 3.300 tasks/domain; absolute deviation: 14.400

Retained historical supplemental exposure IDs: 6847465956a0f6376a605476

### UNUSED (18)

8-9, 15, 23-24, 28, 31-32, 37, 46, 48, 51, 57, 63, 67, 76, 90, 94

## deepsearchqa

Revision: `b2623f8653065c2672de6d941fc5434cd652376c`
Data SHA256: `25d48dcf7efa872e5467032e8b8eedf38d301f59a252d0da95cda584baa78396`
Source: https://huggingface.co/datasets/google/deepsearchqa/resolve/b2623f8653065c2672de6d941fc5434cd652376c/DSQA-full.csv

### Domain Distribution

| Domain | EVO | VAL | TEST |
| --- | ---: | ---: | ---: |
| Arts | 3 | 1 | 4 |
| Arts & Entertainment | 0 | 0 | 1 |
| Biology | 0 | 0 | 2 |
| Current Events | 1 | 0 | 2 |
| Education | 3 | 1 | 3 |
| Finance & Economics | 3 | 0 | 3 |
| Geography | 3 | 1 | 3 |
| Health | 3 | 1 | 4 |
| History | 3 | 1 | 4 |
| Linguistics | 0 | 0 | 1 |
| Media & Entertainment | 3 | 0 | 3 |
| Other | 3 | 1 | 4 |
| Politics & Government | 3 | 1 | 3 |
| Science | 3 | 0 | 3 |
| Sports | 3 | 1 | 3 |
| Technology | 3 | 1 | 3 |
| Travel | 3 | 1 | 4 |

### EVOLUTION (40)

18, 25, 28, 46, 48, 83, 94, 145, 149, 170, 204, 217, 227, 247, 256, 273, 291, 327, 415, 478, 494, 520, 572, 575, 588-589, 593, 604, 639, 662-663, 668, 711, 716, 758, 814, 825, 828, 847, 889

Policy: equal domain quotas within the original parent partition

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 2.353 tasks/domain; absolute deviation: 16.824

Absent from eligible parent pool: Arts & Entertainment; Biology; Linguistics

### VALIDATION (10)

5, 208, 271, 275, 351, 491, 684, 764, 779, 875

Policy: equal domain quotas within the original parent partition

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 0.588 tasks/domain; absolute deviation: 8.235

Available but unrepresented: Finance & Economics; Media & Entertainment; Science

Absent from eligible parent pool: Arts & Entertainment; Biology; Current Events; Linguistics

### TEST (50)

36, 81, 109, 126, 133, 157, 162, 188, 198, 224, 262, 270, 281, 288, 293, 310, 329, 375, 463, 528, 543, 547, 561-562, 579-580, 584, 586, 606, 609, 625, 631, 672, 674-675, 685, 697-698, 700-701, 710, 760, 767, 778, 782, 800, 837, 844, 846, 876

Policy: equal domain quotas within the original parent partition

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 2.941 tasks/domain; absolute deviation: 11.529

### UNUSED (800)

1-4, 6-17, 19-24, 26-27, 29-35, 37-45, 47, 49-80, 82, 84-93, 95-108, 110-125, 127-132, 134-144, 146-148, 150-156, 158-161, 163-169, 171-187, 189-197, 199-203, 205-207, 209-216, 218-223, 225-226, 228-246, 248-255, 257-261, 263-269, 272, 274, 276-280, 282-287, 289-290, 292, 294-309, 311-326, 328, 330-350, 352-374, 376-414, 416-462, 464-477, 479-490, 492-493, 495-519, 521-527, 529-542, 544-546, 548-560, 563-571, 573-574, 576-578, 581-583, 585, 587, 590-592, 594-603, 605, 607-608, 610-624, 626-630, 632-638, 640-661, 664-667, 669-671, 673, 676-683, 686-696, 699, 702-709, 712-715, 717-757, 759, 761-763, 765-766, 768-777, 780-781, 783-799, 801-813, 815-824, 826-827, 829-836, 838-843, 845, 848-874, 877-888, 890-900

## writingbench

Revision: `ae2d5176449b7b769815482641d35926f26793eb`
Data SHA256: `18fee37c645166eb2e206b36366b2e354265b1e4201db2c86e759e825eaddcbe`
Source: https://raw.githubusercontent.com/X-PLUG/WritingBench/ae2d5176449b7b769815482641d35926f26793eb/benchmark_query/benchmark_all.jsonl

### Domain Distribution

| Domain | EVO | VAL | TEST |
| --- | ---: | ---: | ---: |
| Academic & Engineering | 6 | 2 | 8 |
| Advertising & Marketing | 7 | 2 | 9 |
| Education | 7 | 2 | 8 |
| Finance & Business | 6 | 1 | 8 |
| Literature & Arts | 7 | 1 | 8 |
| Politics & Law | 7 | 2 | 9 |

### EVOLUTION (40)

1, 14, 48, 151, 171, 205, 222, 319, 353, 365, 404, 437, 480, 542, 548, 563, 591, 594, 599, 607, 612, 622, 631, 638, 644, 727, 753, 758, 769, 787, 864, 892, 903, 908, 920, 946, 957, 974, 997, 999

Policy: equal primary-domain quotas; balance language within each domain and over the full subset

Languages: en=20, zh=20

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 6.667 tasks/domain; absolute deviation: 2.667

### VALIDATION (10)

24, 227, 268, 286, 348, 411, 432, 525, 696, 711

Policy: equal primary-domain quotas; balance language within each domain and over the full subset

Languages: en=5, zh=5

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 1.667 tasks/domain; absolute deviation: 2.667

### TEST (50)

22, 47, 110, 118, 136, 167, 179, 233, 244, 251, 258-259, 273, 325, 354, 401, 423, 427, 445, 467-468, 568-569, 584, 593, 597, 603, 620, 632, 640-641, 643, 657, 670, 693, 755, 760, 763, 789, 837, 847, 869, 879-880, 967, 970, 982, 984, 986, 995

Policy: equal primary-domain quotas; balance language within each domain and over the full subset

Languages: en=25, zh=25

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 8.333 tasks/domain; absolute deviation: 2.667

### UNUSED (900)

2-13, 15-21, 23, 25-46, 49-109, 111-117, 119-135, 137-150, 152-166, 168-170, 172-178, 180-204, 206-221, 223-226, 228-232, 234-243, 245-250, 252-257, 260-267, 269-272, 274-285, 287-318, 320-324, 326-347, 349-352, 355-364, 366-400, 402-403, 405-410, 412-422, 424-426, 428-431, 433-436, 438-444, 446-466, 469-479, 481-524, 526-541, 543-547, 549-562, 564-567, 570-583, 585-590, 592, 595-596, 598, 600-602, 604-606, 608-611, 613-619, 621, 623-630, 633-637, 639, 642, 645-656, 658-669, 671-692, 694-695, 697-710, 712-726, 728-752, 754, 756-757, 759, 761-762, 764-768, 770-786, 788, 790-836, 838-846, 848-863, 865-868, 870-878, 881-891, 893-902, 904-907, 909-919, 921-945, 947-956, 958-966, 968-969, 971-973, 975-981, 983, 985, 987-994, 996, 998, 1000

## deepresearch_bench_ii

Revision: `b38f360603db9531b102aef8c166cedb8509b6f6`
Data SHA256: `263aaabb8c279fb16cbe7c9499afe82d657a8ab3ccfb07ace084387e367d921a`
Source: https://github.com/imlrz/DeepResearch-Bench-II/tree/b38f360603db9531b102aef8c166cedb8509b6f6

### Domain Distribution

| Domain | EVO | VAL | TEST |
| --- | ---: | ---: | ---: |
| Art & Design | 0 | 0 | 2 |
| Crime & Law | 0 | 0 | 2 |
| Education & Jobs | 0 | 0 | 2 |
| Entertainment | 0 | 0 | 2 |
| Fashion & Beauty | 0 | 0 | 2 |
| Finance & Business | 0 | 0 | 2 |
| Food & Dining | 0 | 0 | 2 |
| Games | 0 | 0 | 2 |
| Hardware | 0 | 0 | 2 |
| Health | 0 | 0 | 2 |
| History | 0 | 0 | 2 |
| Home & Hobbies | 0 | 0 | 2 |
| Industrial | 0 | 0 | 2 |
| Literature | 0 | 0 | 1 |
| Religion | 0 | 0 | 2 |
| Science & Technology | 0 | 0 | 2 |
| Social Life | 0 | 0 | 1 |
| Software | 0 | 0 | 2 |
| Software Development | 0 | 0 | 2 |
| Sports & Fitness | 0 | 0 | 2 |
| Transportation | 0 | 0 | 1 |
| Travel | 0 | 0 | 1 |

### EVOLUTION (0)

none

Policy: equal domain quotas within the original parent partition

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 0.000 tasks/domain; absolute deviation: 0.000

### VALIDATION (0)

none

Policy: equal domain quotas within the original parent partition

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 0.000 tasks/domain; absolute deviation: 0.000

### TEST (40)

12-13, 26, 29, 45, 48, 56, 61, 65-66, 75, 81, 84, 90-91, 93, 96, 98-99, 101-106, 108, 111-114, 117-118, 120, 122-123, 125, 128-129, 131-132

Policy: 20 en and 20 zh; equal theme quotas within each language after known-exposure exclusion

Languages: en=20, zh=20

Absolute deviation from capacity-aware available-domain quotas: 6

Unconstrained ideal over all release domains: 1.818 tasks/domain; absolute deviation: 6.545

### UNUSED (92)

1-11, 14-25, 27-28, 30-44, 46-47, 49-55, 57-60, 62-64, 67-74, 76-80, 82-83, 85-89, 92, 94-95, 97, 100, 107, 109-110, 115-116, 119, 121, 124, 126-127, 130

## ifeval

Revision: `966cd89545d6b6acfd7638bc708b98261ca58e84`
Data SHA256: `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2`
Source: https://huggingface.co/datasets/google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl

### Domain Distribution

| Domain | EVO | VAL | TEST |
| --- | ---: | ---: | ---: |
| all | 0 | 0 | 50 |

### EVOLUTION (0)

none

Policy: no public domain labels; deterministic sampling after known-exposure exclusion

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 0.000 tasks/domain; absolute deviation: 0.000

### VALIDATION (0)

none

Policy: no public domain labels; deterministic sampling after known-exposure exclusion

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 0.000 tasks/domain; absolute deviation: 0.000

### TEST (50)

6, 11, 16, 23, 46, 49, 62, 69, 76, 85, 95, 98, 103, 118, 120, 127, 158, 169, 171, 179, 184, 195, 221, 227, 257, 262, 269, 274, 289, 291, 305, 334, 339, 373, 384, 387, 390, 392, 398-399, 427, 441-442, 456, 463, 498, 502, 506, 522, 529

Policy: no public domain labels; deterministic sampling after known-exposure exclusion

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 50.000 tasks/domain; absolute deviation: 0.000

### UNUSED (491)

1-5, 7-10, 12-15, 17-22, 24-45, 47-48, 50-61, 63-68, 70-75, 77-84, 86-94, 96-97, 99-102, 104-117, 119, 121-126, 128-157, 159-168, 170, 172-178, 180-183, 185-194, 196-220, 222-226, 228-256, 258-261, 263-268, 270-273, 275-288, 290, 292-304, 306-333, 335-338, 340-372, 374-383, 385-386, 388-389, 391, 393-397, 400-426, 428-440, 443-455, 457-462, 464-497, 499-501, 503-505, 507-521, 523-528, 530-541

## ifbench

Revision: `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`
Data SHA256: `d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a`
Source: https://raw.githubusercontent.com/allenai/IFBench/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/data/IFBench_test.jsonl

### Domain Distribution

| Domain | EVO | VAL | TEST |
| --- | ---: | ---: | ---: |
| all | 0 | 0 | 50 |

### EVOLUTION (0)

none

Policy: no public domain labels; deterministic sampling after known-exposure exclusion

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 0.000 tasks/domain; absolute deviation: 0.000

### VALIDATION (0)

none

Policy: no public domain labels; deterministic sampling after known-exposure exclusion

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 0.000 tasks/domain; absolute deviation: 0.000

### TEST (50)

1, 4, 18, 28, 55, 60, 69, 72-73, 75, 77, 82, 100, 108, 116, 123, 125, 127, 136, 151, 153, 155, 162-163, 167, 171-172, 175, 177, 187, 196-198, 205-206, 213, 218, 225, 231, 235, 237, 244, 253, 258, 262-263, 267, 277, 289-290

Policy: no public domain labels; deterministic sampling after known-exposure exclusion

Absolute deviation from capacity-aware available-domain quotas: 0

Unconstrained ideal over all release domains: 50.000 tasks/domain; absolute deviation: 0.000

### UNUSED (250)

2-3, 5-17, 19-27, 29-54, 56-59, 61-68, 70-71, 74, 76, 78-81, 83-99, 101-107, 109-115, 117-122, 124, 126, 128-135, 137-150, 152, 154, 156-161, 164-166, 168-170, 173-174, 176, 178-186, 188-195, 199-204, 207-212, 214-217, 219-224, 226-230, 232-234, 236, 238-243, 245-252, 254-257, 259-261, 264-266, 268-276, 278-288, 291-300

## Independent Evolution Batches

Each source/run starts from zero; five tasks share one state before three candidates are produced.

### Run 0 (seed 20261001)

#### researchrubrics

Batch 1: 42, 79, 38, 74, 61

Batch 2: 62, 88, 59, 82, 6

Batch 3: 89, 1, 53, 14, 41

Batch 4: 66, 98, 92, 13, 20

Batch 5: 45, 54, 40, 96, 2

Batch 6: 17, 84, 75, 34, 68

Batch 7: 52, 86, 64, 70, 35

Batch 8: 60, 21, 101, 12, 100

#### deepsearchqa

Batch 1: 575, 46, 588, 256, 847

Batch 2: 589, 478, 291, 217, 668

Batch 3: 520, 327, 639, 716, 28

Batch 4: 825, 18, 662, 94, 48

Batch 5: 25, 572, 247, 758, 814

Batch 6: 83, 711, 593, 604, 494

Batch 7: 170, 149, 273, 415, 889

Batch 8: 227, 204, 145, 828, 663

#### writingbench

Batch 1: 908, 171, 404, 14, 753

Batch 2: 946, 319, 997, 353, 48

Batch 3: 622, 563, 727, 1, 758

Batch 4: 205, 542, 151, 594, 920

Batch 5: 607, 644, 892, 787, 769

Batch 6: 480, 631, 903, 437, 222

Batch 7: 548, 599, 974, 591, 638

Batch 8: 365, 864, 957, 612, 999

### Run 1 (seed 20261002)

#### researchrubrics

Batch 1: 2, 45, 34, 59, 13

Batch 2: 40, 61, 101, 17, 100

Batch 3: 70, 62, 96, 41, 82

Batch 4: 75, 38, 74, 14, 1

Batch 5: 53, 88, 79, 52, 42

Batch 6: 86, 68, 35, 20, 64

Batch 7: 92, 12, 54, 89, 60

Batch 8: 84, 66, 6, 21, 98

#### deepsearchqa

Batch 1: 94, 273, 847, 227, 593

Batch 2: 662, 711, 18, 149, 327

Batch 3: 204, 825, 639, 145, 663

Batch 4: 589, 415, 572, 291, 478

Batch 5: 814, 247, 604, 668, 828

Batch 6: 46, 520, 575, 758, 588

Batch 7: 83, 217, 48, 494, 256

Batch 8: 28, 25, 170, 716, 889

#### writingbench

Batch 1: 997, 548, 638, 599, 908

Batch 2: 974, 205, 903, 946, 644

Batch 3: 957, 753, 769, 612, 892

Batch 4: 404, 319, 365, 727, 542

Batch 5: 437, 631, 787, 758, 1

Batch 6: 353, 151, 171, 594, 480

Batch 7: 48, 607, 864, 920, 222

Batch 8: 591, 622, 999, 563, 14

### Run 2 (seed 20261003)

#### researchrubrics

Batch 1: 70, 62, 12, 74, 17

Batch 2: 79, 52, 86, 100, 45

Batch 3: 2, 98, 53, 66, 64

Batch 4: 1, 68, 21, 20, 41

Batch 5: 59, 40, 60, 42, 54

Batch 6: 96, 92, 82, 34, 101

Batch 7: 6, 89, 13, 35, 38

Batch 8: 88, 84, 14, 61, 75

#### deepsearchqa

Batch 1: 639, 291, 204, 46, 170

Batch 2: 589, 18, 415, 711, 825

Batch 3: 520, 716, 256, 227, 247

Batch 4: 217, 758, 663, 28, 494

Batch 5: 478, 662, 327, 25, 889

Batch 6: 604, 572, 149, 575, 593

Batch 7: 847, 814, 94, 588, 145

Batch 8: 48, 83, 668, 828, 273

#### writingbench

Batch 1: 920, 644, 594, 319, 908

Batch 2: 480, 548, 222, 591, 946

Batch 3: 957, 48, 404, 171, 769

Batch 4: 638, 727, 607, 563, 997

Batch 5: 1, 999, 631, 437, 612

Batch 6: 892, 542, 14, 151, 974

Batch 7: 622, 903, 864, 205, 353

Batch 8: 753, 787, 365, 758, 599
