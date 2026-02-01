import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 1. Load the Data
csv_data = """Features,Experiment_Name,Eval_Mode,HR@10,HR@100,HR@10_new,HR@100_new,Timestamp
Both features,70-30 Split (Items),Rolling (No Mask),0.01912925565402683,0.09375756695239479,0.007002561912894962,0.049758041559920294,2026-01-04 01:11:13
Both features,70-30 Split (Items),Rolling (Availability Mask),0.023100392270812144,0.11448496295220108,0.009052092228864219,0.06450327355536578,2026-01-04 01:12:19
Both features,70-30 Split (Items),Static (Pure),0.0868225019265348,0.25404572309273055,0.02468487394957983,0.1722689075630252,2026-01-04 01:12:28
Both features,Leave-One-Out (Items),Rolling (No Mask),0.030310814282044695,0.10043668122270742,0.011624688624411846,0.06531967893717133,2026-01-04 01:12:37
Both features,Leave-One-Out (Items),Rolling (Availability Mask),0.03390701258669407,0.11816080143847932,0.013562136728480487,0.08137282037088292,2026-01-04 01:12:53
Both features,Leave-One-Out (Items),Static (Pure),0.030310814282044695,0.10043668122270742,0.011624688624411846,0.06531967893717133,2026-01-04 01:13:01
Both features,70-30 Split (Groups),Static (Pure),0.0963020030816641,0.288135593220339,0.033079548437910215,0.22709372538724074,2026-01-04 01:13:10
Both features,Leave-One-Out (Groups),Static (Pure),0.03826399589111454,0.14227015921931177,0.019092418372993913,0.10736026563364692,2026-01-04 01:13:19
Image features,70-30 Split (Items),Rolling (No Mask),0.020049397065233183,0.09947212940093951,0.00631938514090521,0.05397096498719044,2026-01-04 01:14:03
Image features,70-30 Split (Items),Rolling (Availability Mask),0.025570245532471305,0.11966681195215265,0.00865357244520353,0.06939937375462568,2026-01-04 01:15:03
Image features,70-30 Split (Items),Static (Pure),0.09118931415360904,0.26303621885435396,0.025735294117647058,0.1796218487394958,2026-01-04 01:15:12
Image features,Leave-One-Out (Items),Rolling (No Mask),0.02337528898022091,0.10403287952735679,0.009687240520343206,0.0763908109604207,2026-01-04 01:15:21
Image features,Leave-One-Out (Items),Rolling (Availability Mask),0.026714615977395325,0.12175699974312869,0.011624688624411846,0.0916136174923886,2026-01-04 01:15:37
Image features,Leave-One-Out (Items),Static (Pure),0.02337528898022091,0.10403287952735679,0.009687240520343206,0.0763908109604207,2026-01-04 01:15:46
Image features,70-30 Split (Groups),Static (Pure),0.10349255264509502,0.3017462763225475,0.03045418745077448,0.23523234444736152,2026-01-04 01:15:55
Image features,Leave-One-Out (Groups),Static (Pure),0.035182331792501284,0.14612223934257831,0.015218594355285003,0.11040398450470393,2026-01-04 01:16:03
Tag features,70-30 Split (Items),Rolling (No Mask),0.020243111046539784,0.0965664196813405,0.00825505266154284,0.05550811272416738,2026-01-04 01:16:47
Tag features,70-30 Split (Items),Rolling (Availability Mask),0.025037532083878152,0.11613153179330718,0.01058923996584116,0.06939937375462568,2026-01-04 01:17:47
Tag features,70-30 Split (Items),Static (Pure),0.08425378885178525,0.26868738761880295,0.027836134453781514,0.19170168067226892,2026-01-04 01:17:55
Tag features,Leave-One-Out (Items),Rolling (No Mask),0.0279989725147701,0.11045466221423067,0.01190146692499308,0.0772211458621644,2026-01-04 01:18:04
Tag features,Leave-One-Out (Items),Rolling (Availability Mask),0.03056768558951965,0.12715129720010274,0.013838915029061722,0.09189039579296983,2026-01-04 01:18:20
Tag features,Leave-One-Out (Items),Static (Pure),0.0279989725147701,0.11045466221423067,0.01190146692499308,0.0772211458621644,2026-01-04 01:18:28
Tag features,70-30 Split (Groups),Static (Pure),0.09784283513097072,0.2794042116076014,0.03255447624048307,0.21632974533998425,2026-01-04 01:18:36
Tag features,Leave-One-Out (Groups),Static (Pure),0.0362095531587057,0.13533641499743196,0.015771997786386275,0.09684560044272275,2026-01-04 01:18:45
No features,70-30 Split (Items),Rolling (No Mask),0.016514116906387717,0.08813986149450337,0.006376316538571022,0.050156561343580985,2026-01-04 01:19:26
No features,70-30 Split (Items),Rolling (Availability Mask),0.02111482396241949,0.10983582740084266,0.008311984059208653,0.06552803871335042,2026-01-04 01:20:24
No features,70-30 Split (Items),Static (Pure),0.07629077832006165,0.2504495247880812,0.020483193277310924,0.17358193277310924,2026-01-04 01:20:32
No features,Leave-One-Out (Items),Rolling (No Mask),0.025687130747495505,0.09684048291805805,0.008856905618599501,0.07085524494879601,2026-01-04 01:20:39
No features,Leave-One-Out (Items),Rolling (Availability Mask),0.02902645774466992,0.11071153352170562,0.011347910323830611,0.08247993357320786,2026-01-04 01:20:55
No features,Leave-One-Out (Items),Static (Pure),0.025687130747495505,0.09684048291805805,0.008856905618599501,0.07085524494879601,2026-01-04 01:21:02
No features,70-30 Split (Groups),Static (Pure),0.108371854134566,0.29198767334360554,0.03045418745077448,0.2168548175374114,2026-01-04 01:21:10
No features,Leave-One-Out (Groups),Static (Pure),0.036466358500256806,0.13687724704673856,0.016048699501936912,0.09767570558937465,2026-01-04 01:21:18
"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. Setup the Plotting
# We can loop through metrics, but let's just plot the most important one: HR@100
# (You can change y_metric to 'HR@10', 'HR@100', 'HR@10_new', or 'HR@100_new')
y_metric = 'HR@10'

sns.set_theme(style="whitegrid")

# Create a FacetGrid: Rows = Feature Sets
g = sns.catplot(
    data=df,
    x='Experiment_Name',
    y=y_metric,
    hue='Eval_Mode',
    col='Features',       # Create a separate chart for each Feature type
    col_wrap=2,           # 2 charts per row (2x2 grid)
    kind='bar',
    height=4,
    aspect=1.6,
    palette='magma'       # 'magma', 'viridis', or 'deep' look good
)

# 3. Polish the Layout
g.fig.suptitle(f'Comparison of {y_metric} Across Feature Sets', y=1.02, fontsize=16, fontweight='bold')

# Loop through axes to clean up labels
for ax in g.axes.flat:
    # Rotate x-axis labels
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=2, fontsize=8, rotation=90)

# Move Legend
sns.move_legend(
    g, "upper left",
    bbox_to_anchor=(0, 1),
    title='Evaluation Mode',
    frameon=True
)

plt.tight_layout()
plt.show()