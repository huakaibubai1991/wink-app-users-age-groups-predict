#!/bin/bash

#####################################################################
# author: zxf3@meitu.com
#
# 用于wink年龄段模型预测
#
#####################################################################

# 执行参数校验
# 无参数时统计前一天的数据
# 单一参数时统计指定日期的数据

if [ $# -eq 0 ]; then
    load_date=$(date -d "-1 day" +"%Y%m%d")
elif [ $# -eq 1 ]; then
    load_date=$1
else
    echo "args error: load_date"
    exit 1
fi
echo $load_date

####################################################################
# 脚本初始化
####################################################################

HOME=$(cd "$(dirname "$0")" || exit; pwd)
cd "$HOME" || exit;
echo $HOME
HOME_UP=`cd ..; pwd`
echo $HOME_UP

prepare_data(){

    output_path_predict="$HOME_UP/data/query_result_predict_data-${load_date}.txt"
    echo $output_path_predict

    #### 处理SQL文件中的参数
    predict_data_sql_prepare=`cat $HOME_UP/sql/wink_agestage_test_data.sql`
    predict_data_sql_final=${predict_data_sql_prepare//'day_param'/$load_date}
    echo $predict_data_sql_final

    echo "------------------------start predict data sql----------------------------"

    #### 执行SQL查出训练集数据
    datawork-client  query_to_local  -hql "$predict_data_sql_final"  -project_name userprofile    -ca_config_path /data1/www/zxf3/datawork_clinet_conf/ca_config_user_profile_mid.properties -file_path ${output_path_predict} -env prd -hive_env huawei -v

    echo "------------------------end predict data sql-----------------------------"
}


process_data(){

    input_data="$HOME_UP/data/query_result_predict_data-${load_date}.txt"
    output_data_libsvm="$HOME_UP/data/process_predict_data_libsvm-${load_date}.txt"
    input_data_split="$HOME_UP/data/query_result_predict_data-${load_date}-split"
    output_data_libsvm_split="$HOME_UP/data/process_predict_data_libsvm-${load_date}-split"

    #### 分割测试数据
    split -l 2000000 $input_data -d -a 1 $input_data_split

    #### 更换目录
    cd "$HOME_UP/data"

    input_data_split_cnt=`ls -l | grep query_result_predict_data-${load_date}-split | wc -l`
    input_data_split_cnt_final=`expr $input_data_split_cnt - 1`
    echo $input_data_split_cnt
    echo $input_data_split_cnt_final

    #### 将TXT格式的数据转换为libsvm格式的数据
    echo "-----------------------start transform_txt_to_libsvm process---------------------------"

    if [ -f "$input_data" ]
    then
        for data in `seq 0 $input_data_split_cnt_final`;do
            echo $data
            echo $input_data_split$data $output_data_libsvm_split$data
            /usr/local/python3/bin/python3 $HOME/trans_txt_to_libsvm_predict.py $input_data_split$data $output_data_libsvm_split$data
        done
    else
        echo "数据集为空"
    fi

    #/home/hadoopuser/anaconda3/bin/python $HOME/trans_txt_to_libsvm_predict.py $input_data $output_data_libsvm
    echo "-----------------------end transform_txt_to_libsvm process---------------------------"

    #### 将文件合并
    cat $output_data_libsvm_split* > $output_data_libsvm

    #### 删除split的数据
    rm -rf $input_data_split*
    rm -rf $output_data_libsvm_split*

}


predict_process(){

    input_data_predict="$HOME_UP/data/process_predict_data_libsvm-${load_date}.txt"
    model_path="$HOME_UP/wink_agestage_train_lgbm_202410.model"
    result_path="$HOME_UP/data/predict_result_data-${load_date}.txt"
    result_path_gid="$HOME_UP/data/predict_result_data_trans_gid-${load_date}.txt"
    result_path_final="$HOME_UP/data/predict_result_data_final-${load_date}.txt"

    #### 模型预测
    echo "-----------------------start predict process---------------------------"
    /usr/local/python3/bin/python3 $HOME/wink_agestage_predict_model.py $input_data_predict $model_path $result_path
    echo "------------------------end predict process----------------------------"

    #### 处理模型预测之后17位gid转换为科学计数法的问题
    awk 'NR==FNR{a[NR]=$1; next} {$1=a[FNR]}1' $HOME_UP/data/query_result_predict_data-${load_date}.txt $HOME_UP/data/predict_result_data-${load_date}.txt > $HOME_UP/data/predict_result_data_trans_gid-${load_date}.txt

    #### 处理之后的分隔符问题
    awk '{print $1 "\t" $2 "\t" substr($0, index($0, $3))}' $result_path_gid > $result_path_final
    rm -rf $HOME_UP/data/predict_result_data_trans_gid-${load_date}.txt

    #### 将产出数据导入hive表
    datawork-client execute -hql  "load data local inpath $result_path_final  overwrite into table user_profile_mid.uprofile_odz_agestage_model_predict_probability partition(date_p=${load_date},type_p=\"wink_v1\")"  -project_name userprofile  -ca_config_path /data1/www/zxf3/datawork_clinet_conf/ca_config_user_profile_mid.properties -env prd  -hive_env huawei -v

}

clear_log(){

    clear_date=$(date -d "${load_date} - 1 day" +%Y%m%d)
    echo $clear_date
    rm -rf $HOME_UP/data/query_result_predict_data-${clear_date}.txt
    rm -rf $HOME_UP/data/process_predict_data_libsvm-${clear_date}.txt
    rm -rf $HOME_UP/data/predict_result_data-${clear_date}.txt
}


main(){

    prepare_data
    process_data
    predict_process
    clear_log
}

main

