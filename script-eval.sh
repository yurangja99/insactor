export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1

# run evaluation for
# 1. KIT 0.01
# 2. KIT 0.01 (perturb)
# 3. KIT 0.01 (waypoint)
# 4. HumanML3D 0.01
# 5. HumanML3D 0.01 (perturb)
# 6. HumanML3D 0.01 (waypoint)

# eval names
name_list=(
    "KIT 0.01"
    "KIT 0.01 (perturb)"
    "KIT 0.01 (waypoint)"
    "HumanML3D 0.01"
    "HumanML3D 0.01 (perturb)"
    "HumanML3D 0.01 (waypoint)"
)

# controller param paths
controller_path_list=(
    "pretrained_models/skill_kit_0.01.pkl"
    "pretrained_models/skill_kit_0.01.pkl"
    "pretrained_models/skill_kit_0.01.pkl"
    "pretrained_models/skill_human_0.01.pkl"
    "pretrained_models/skill_human_0.01.pkl"
    "pretrained_models/skill_human_0.01.pkl"
)

# eval result dir
work_dir_list=(
    "kit" 
    "kit_perturb" 
    "kit_waypoint" 
    "human" 
    "human_perturb" 
    "human_waypoint"
)

# command list
cmd_list=(
    "python -u tools/test.py configs/planner/kit.py --physmode=normal pretrained_models/planner_kit.pth"
    "python -u tools/test.py configs/planner/kit.py --physmode=normal --perturb true pretrained_models/planner_kit.pth"
    "python -u tools/test_waypoint.py configs/planner/kit.py --physmode=normal pretrained_models/planner_kit.pth"
    "python -u tools/test.py configs/planner/human.py --physmode=normal pretrained_models/planner_humanml.pth"
    "python -u tools/test.py configs/planner/human.py --physmode=normal --perturb true pretrained_models/planner_humanml.pth"
    "python -u tools/test_waypoint.py configs/planner/human.py --physmode=normal pretrained_models/planner_humanml.pth"
)

# make work directory
if [ -d "work_dirs" ]; then
    :
else
    mkdir "work_dirs"
fi
store_dir="work_dirs/eval_$(date +'%Y%m%d%H%M%S')"
mkdir "${store_dir}/"
echo "store evaluation result at: ${store_dir}/"

# run evaluations
for i in {0..5}
do
    echo "============================================================="
    echo "Evaluation $i: ${name_list[$i]}"
    # set controller parameter path
    export CONTROLLER_PARAM_PATH="${controller_path_list[$i]}"
    # make store directory
    mkdir "${store_dir}/${work_dir_list[$i]}"
    # evaluation
    echo -n "start evaluation ... "
    echo "Evaluation ${name_list[$i]}" >> "${store_dir}/${work_dir_list[$i]}/log"
    echo "Controller ${controller_path_list[$i]}" >> "${store_dir}/${work_dir_list[$i]}/log"
    eval "${cmd_list[$i]} --work-dir=${store_dir} | tee -a ${store_dir}/${work_dir_list[$i]}/log"
    awk '!/\[/' "${store_dir}/${work_dir_list[$i]}/log" >> "${store_dir}/${work_dir_list[$i]}/result"
    echo "done!"
done
