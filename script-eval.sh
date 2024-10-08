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
    "work_dirs/eval/kit" 
    "work_dirs/eval/kit_perturb" 
    "work_dirs/eval/kit_waypoint" 
    "work_dirs/eval/human" 
    "work_dirs/eval/human_perturb" 
    "work_dirs/eval/human_waypoint"
)

# command list
cmd_list=(
    "python -u tools/test.py configs/planner/kit.py --work-dir=work_dirs/eval --physmode=normal pretrained_models/planner_kit.pth"
    "python -u tools/test.py configs/planner/kit.py --work-dir=work_dirs/eval --physmode=normal --perturb true pretrained_models/planner_kit.pth"
    "python -u tools/test_waypoint.py configs/planner/kit.py --work-dir=work_dirs/eval --physmode=normal pretrained_models/planner_kit.pth"
    "python -u tools/test.py configs/planner/human.py --work-dir=work_dirs/eval --physmode=normal pretrained_models/planner_humanml.pth"
    "python -u tools/test.py configs/planner/human.py --work-dir=work_dirs/eval --physmode=normal --perturb true pretrained_models/planner_humanml.pth"
    "python -u tools/test_waypoint.py configs/planner/human.py --work-dir=work_dirs/eval --physmode=normal pretrained_models/planner_humanml.pth"
)

# make work directory
if [ -d "work_dirs" ]; then
    echo "work_dirs already exists"
else
    echo "create work_dirs directory"
    mkdir "work_dirs"
fi

if [ -d "work_dirs/eval" ]; then
    echo "work_dirs/eval already exists"
else
    echo "create work_dirs/eval directory"
    mkdir "work_dirs/eval"
fi

# run evaluations
for i in 0 1 2 3 4 5
do
    echo "============================================================="
    echo "Evaluation $i: ${name_list[$i]}"
    # set controller parameter path
    export CONTROLLER_PARAM_PATH="${controller_path_list[$i]}"
    # rename existing evaluation result directory
    if [ -d "${work_dir_list[$i]}" ]; then
        num=0
        while [ -d "${work_dir_list[$i]}_${num}" ]; do
            ((num++))
        done
        mv "${work_dir_list[$i]}" "${work_dir_list[$i]}_${num}"
        echo "rename existing work directory to: ${work_dir_list[$i]}_${num}"
    else
        echo "create work directory: ${work_dir_list[$i]}"
    fi
    mkdir "${work_dir_list[$i]}"
    # evaluation
    echo -n "start evaluation ... "
    echo "Evaluation ${name_list[$i]}" >> "${work_dir_list[$i]}/log.out"
    echo "Controller ${controller_path_list[$i]}" >> "${work_dir_list[$i]}/log.out"
    eval "${cmd_list[$i]} >> ${work_dir_list[$i]}/log.out 2>> ${work_dir_list[$i]}/log.err"
    echo "done!"
done
