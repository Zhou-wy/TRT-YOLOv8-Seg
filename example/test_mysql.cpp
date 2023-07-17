/**
 * brief: mysql简单测试
 */
#include "../src/SqlWarpper/mysql.h"

/**
 * brief: 基础增删改查功能测试(有数据会超长插失败, 属于正常现象)
 * details: 自己新建一张表
 *          表名: stduent
 *          列:   code    char(10)
 *                name    varchar(60)
 *                age     int
 *                sex     char(1)
 *                time    timestamp - 默认值设置为CURRENT_TIMESTAMP, 可为null, 设置为自动更新, 不用每次自己去插时间
 */

/**

CREATE TABLE VideoDetectResult (
    ID BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    Site VARCHAR(255),
    Date DATETIME,
    Result VARCHAR(64),
    Keyframe VARCHAR(255)
);

ALTER TABLE VideoDetectResult MODIFY COLUMN Site VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;


 */

void test_base()
{
    std::cout << "----------增删改查功能测试----------\n";
#define XX(name, cmd)                                                  \
    {                                                                  \
        mysql::MySQLRes::ptr res = MySQLUtil::query(name, cmd);        \
        if (res)                                                       \
        {                                                              \
            std::cout << "总行数: " << res->getRows() << std::endl;     \
            const std::vector<std::string> &fields = res->getFields(); \
            for (auto field : fields)                                  \
                std::cout << field << "\t";                            \
            std::cout << std::endl;                                    \
            while (res->next())                                        \
            {                                                          \
                try                                                    \
                {                                                      \
                    for (auto field : fields)                          \
                        std::cout << res->getString(field) << "\t";    \
                }                                                      \
                catch (std::string & ex)                               \
                {                                                      \
                    std::cout << ex << std::endl;                      \
                }                                                      \
                std::cout << std::endl;                                \
            }                                                          \
        }                                                              \
    }

    MySQLUtil::execute("suep_echarger", "insert into VideoDetectResult (ID, Site, Date, Result, Keyframe) values (%d, '%s', '%s', '%s', '%s')", 0, "杨树浦路", "20230706", "normal", "./images/1.jpg");

    std::cout << "更新前查询" << std::endl;
    XX("suep_echarger", "select * from VideoDetectResult");
    std::cout << std::endl;

    // 更新刚刚插入的数据
    MySQLUtil::execute("suep_echarger", "UPDATE VideoDetectResult SET  Site = '%s', Date = '%s', Result = '%s', Keyframe = '%s' WHERE ID = %d ", "杨树浦路", "20230706", "normal", "./images/1.jpg", 0);

    std::cout << std::endl;

    std::cout << "更新后查询" << std::endl;
    std::string select_sql = "select * from VideoDetectResult";
    XX("suep_echarger", select_sql.c_str());
    std::cout << std::endl;

    // 删除表内容
#if 0
    for (int i = 0; i < 10; i++)
        MySQLUtil::execute("sql", "delete from VideoDetectResult where code = '%s'", code[i]);
    std::cout << std::endl;

    std::cout << "删除后查询" << std::endl;
    XX("sql", "select * from VideoDetectResult");
    std::cout << std::endl;
#endif

#undef XX
}

/**
 * brief: 预处理功能测试
 */
void test_stmt()
{
    std::cout << "----------预处理功能测试----------\n";
#define XX(is_convert)                                                       \
    {                                                                        \
        mysql::MySQLStmtRes::ptr stmt_res = stmt->query();                   \
        std::cout << "总行数: " << stmt_res->getRows() << std::endl;         \
        const std::vector<std::string> &fields = stmt_res->getFields();      \
        for (auto field : fields)                                            \
            std::cout << field << "\t";                                      \
        std::cout << std::endl;                                              \
        while (stmt_res->next())                                             \
        {                                                                    \
            for (auto field : fields)                                        \
                std::cout << stmt_res->getString(field, is_convert) << "\t"; \
            std::cout << std::endl;                                          \
        }                                                                    \
    }

    mysql::MySQLStmt::ptr stmt = MySQLUtil::openPrepare("sql", "select * from VideoDetectResult where sex = ?");

    std::cout << "查询男生信息" << std::endl;
    stmt->multibind("Y");
    XX(true);
    std::cout << std::endl;

    std::cout << "查询女生信息" << std::endl;
    stmt->multibind("N");
    XX(false);
    std::cout << std::endl;

#undef XX
}

/**
 * brief: 事务功能测试
 */
void test_transaction()
{
    std::cout << "----------事务功能测试----------\n";
#define XX(age, auto_commit)                                                                          \
    {                                                                                                 \
        mysql::MySQLTransaction::ptr transaction = MySQLUtil::openTransaction("sql", auto_commit);    \
        transaction->begin();                                                                         \
        transaction->execute("update VideoDetectResult set age = %d where code = '1234567890'", age); \
    }

#define XXX(cmd)                                                       \
    {                                                                  \
        mysql::MySQLRes::ptr res = MySQLUtil::query("sql", cmd);       \
        if (res)                                                       \
        {                                                              \
            std::cout << "总行数: " << res->getRows() << std::endl;    \
            const std::vector<std::string> &fields = res->getFields(); \
            for (auto field : fields)                                  \
                std::cout << field << "\t";                            \
            std::cout << std::endl;                                    \
            while (res->next())                                        \
            {                                                          \
                try                                                    \
                {                                                      \
                    for (auto field : fields)                          \
                        std::cout << res->getString(field) << "\t";    \
                }                                                      \
                catch (std::string ex)                                 \
                {                                                      \
                    std::cout << ex << std::endl;                      \
                }                                                      \
                std::cout << std::endl;                                \
            }                                                          \
        }                                                              \
    }

    std::cout << "执行事务之后提交" << std::endl;
    std::cout << "第一次查询" << std::endl;
    XXX("select * from VideoDetectResult");
    XX(100, true);
    std::cout << "第二次查询" << std::endl;
    XXX("select * from VideoDetectResult");

    std::cout << std::endl;

    std::cout << "执行事务之后回滚" << std::endl;
    std::cout << "第一次查询" << std::endl;
    XXX("select * from VideoDetectResult");
    XX(50, false);
    std::cout << "第二次查询" << std::endl;
    XXX("select * from VideoDetectResult");

#undef XX
#undef XXX
}

int main(int argc, char *argv[])
{
    mysql::MySQLMgr::GetInstance()->add("suep_echarger", "127.0.0.1", 3306, "root", "12345678", "eCharger");
    test_base();
    // test_stmt();
    // test_transaction();
    return 0;
}
