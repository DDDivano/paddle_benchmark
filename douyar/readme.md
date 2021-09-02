# paddle vs torch 前反向对比工具
参考test.py文件
大概分成4步：
1. 实例化一个Douyar对象，传入需要测试的api
2. 初始化需要的入参，使用dict配置
3. set_xxx_param传入入参
4. obj.run 查看结果