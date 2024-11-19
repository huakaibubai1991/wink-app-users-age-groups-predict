#!/bin/bash

#####################################################################
# author: zxf3@meitu.com
#
# 用于wink年龄段模型训练
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

    output_path_train="$HOME_UP/data/query_result_train_data-${load_date}.txt"
    echo $output_path_train

    #### 处理SQL文件中的参数
    train_data_sql_prepare=`cat $HOME_UP/sql/wink_agestage_train_data.sql`
    train_data_sql_final=${train_data_sql_prepare//'day_param'/$load_date}
    echo $train_data_sql_final

    echo "------------------------start train data sql----------------------------"

    #### 执行SQL查出训练集数据
    datawork-client  query_to_local  -hql "$train_data_sql_final"  -project_name userprofile    -ca_config_path /data1/www/zxf3/datawork_clinet_conf/ca_config_user_profile_mid.properties -file_path ${output_path_train} -env prd -hive_env huawei -v

    echo "------------------------end train data sql-----------------------------"
}


process_data(){

    input_data="$HOME_UP/data/query_result_train_data-${load_date}.txt"
    output_data_libsvm="$HOME_UP/data/process_train_data_libsvm-${load_date}.txt"
    
    #### 将TXT格式的数据转换为libsvm格式的数据
    echo "-----------------------start transform_txt_to_libsvm process---------------------------"   
    /usr/local/python3/bin/python3 $HOME/trans_txt_to_libsvm.py $input_data $output_data_libsvm
    echo "-----------------------end transform_txt_to_libsvm process---------------------------"

}


train_process(){

    input_data_train="$HOME_UP/data/process_train_data_libsvm-${load_date}.txt"
    model_path="$HOME_UP/wink_agestage_train_lgbm.model"
    
    #### 模型训练
    echo "-----------------------start train process---------------------------"
    /usr/local/python3/bin/python3 $HOME/wink_agestage_train_model.py $input_data_train $model_path
    echo "------------------------end train process----------------------------"

    #### 模型备份
    cp $HOME_UP/wink_agestage_train_lgbm.model $HOME_UP/wink_agestage_train_lgbm_${load_date:0:6}.model
}


clear_log(){

    clear_date=$(date -d "${load_date} - 1 day" +%Y%m%d)
    echo $clear_date
    rm -rf $HOME_UP/data/query_result_train_data-${clear_date}.txt
    rm -rf $HOME_UP/data/process_train_data_libsvm-${clear_date}.txt
}


main(){

    prepare_data
    process_data
    train_process
    clear_log
}

main


