#include "mysql.h"

namespace mysql
{

namespace
{
    struct MySQLThreadInit
    {
        MySQLThreadInit()
        {
            mysql_thread_init();
        }

        ~MySQLThreadInit()
        {
            mysql_thread_end();
        }
    };
}

std::string time_to_string(time_t ts, const std::string& format, bool is_gmt)
{
    struct tm t;
    if (is_gmt)
        gmtime_r(&ts, &t);
    else
        localtime_r(&ts, &t);
    char buf[64] = {0};
    strftime(buf, sizeof(buf), format.c_str(), &t);
    return buf;
}

std::string mysql_bind_to_string(const MYSQL_BIND& bind)
{
    std::stringstream ss;

#define XX(type, _type, u_type) \
    case type: \
        ss << "buffer_type=" << #type \
           << "\t" "is_unsigned=" << (bind.is_unsigned ? "true" : "false") \
           << "\t" "buffer_length=" << bind.buffer_length \
           << "\t" "buffer=" << (bind.is_unsigned ? *(u_type*)bind.buffer : *(_type*)bind.buffer); \
        break;

#define XXX(type) \
    case type: \
        ss << "buffer_type=" << #type \
           << "\t" "buffer_length=" << bind.buffer_length \
           << "\t" "buffer=" << std::string((char*)bind.buffer, bind.buffer_length); \
        break;

#define XXXX(type) \
    case type: \
        ss << "buffer_type=" << #type \
           << "\t" "buffer_length=" << bind.buffer_length \
           << "\t" "buffer=" << time_to_string(mysql_time_to_time_t(*(MYSQL_TIME*)bind.buffer)); \
        break;

    switch (bind.buffer_type)
    {
        XXX(MYSQL_TYPE_DECIMAL);
        XX(MYSQL_TYPE_TINY, int8_t, uint8_t);
        XX(MYSQL_TYPE_SHORT, int16_t, uint16_t);
        XX(MYSQL_TYPE_LONG, int32_t, uint32_t);
        XX(MYSQL_TYPE_FLOAT, float, float);
        XX(MYSQL_TYPE_DOUBLE, double, double);
        XXX(MYSQL_TYPE_NULL);
        XXXX(MYSQL_TYPE_TIMESTAMP);
        XX(MYSQL_TYPE_LONGLONG, int64_t, uint64_t);
        XXX(MYSQL_TYPE_INT24);
        XXXX(MYSQL_TYPE_DATE);
        XXXX(MYSQL_TYPE_TIME);
        XXXX(MYSQL_TYPE_DATETIME);
        XXX(MYSQL_TYPE_YEAR);
        XXX(MYSQL_TYPE_NEWDATE);
        XXX(MYSQL_TYPE_VARCHAR);
        XXX(MYSQL_TYPE_BIT);
        XXX(MYSQL_TYPE_NEWDECIMAL);
        XXX(MYSQL_TYPE_ENUM);
        XXX(MYSQL_TYPE_SET);
        XXX(MYSQL_TYPE_TINY_BLOB);
        XXX(MYSQL_TYPE_MEDIUM_BLOB);
        XXX(MYSQL_TYPE_LONG_BLOB);
        XXX(MYSQL_TYPE_BLOB);
        XXX(MYSQL_TYPE_VAR_STRING);
        XXX(MYSQL_TYPE_STRING);
        XXX(MYSQL_TYPE_GEOMETRY);

        default:
            break;
    }

    return ss.str();

#undef XX
#undef XXX
#undef XXXX
}

time_t mysql_time_to_time_t(const MYSQL_TIME& mt)
{
    struct tm tm;
    memset(&tm, 0x00, sizeof(tm));
    tm.tm_year = mt.year - 1900;
    tm.tm_mon = mt.month - 1;
    tm.tm_mday = mt.day;
    tm.tm_hour = mt.hour;
    tm.tm_min = mt.minute;
    tm.tm_sec = mt.second;
    time_t t = mktime(&tm);
    return t < 0 ? 0 : t;
}

MYSQL_TIME time_t_to_mysql_time(const time_t& ts)
{
    struct tm tm;
    localtime_r(&ts, &tm);
    MYSQL_TIME mt;
    memset(&mt, 0x00, sizeof(mt));
    mt.year = tm.tm_year + 1900;
    mt.month = tm.tm_mon + 1;
    mt.day = tm.tm_mday;
    mt.hour = tm.tm_hour;
    mt.minute = tm.tm_min;
    mt.second = tm.tm_sec;
    return mt;
}


// static
MySQLStmtRes::ptr MySQLStmtRes::create(std::shared_ptr<MySQLStmt> stmt)
{
    if (stmt->getErrno())
    {
        std::cout << "stmt error, errno=" << stmt->getErrno()
            << ", errstr=" << stmt->getErrstr() << std::endl;
        return nullptr;
    }
    MySQLStmtRes::ptr ret(new MySQLStmtRes(stmt));

    MYSQL_RES* res = mysql_stmt_result_metadata(stmt->get());
    if (!res)
    {
        std::cout << "mysql_stmt_result_metadata error, errno="
            << stmt->getErrno() << ", errstr=" << stmt->getErrstr() << std::endl;
        return nullptr;
    }
    int len = mysql_num_fields(res);
    MYSQL_FIELD* fields = mysql_fetch_fields(res);
    ret->m_binds.resize(len);
    memset(&ret->m_binds[0], 0, sizeof(ret->m_binds[0]) * len);

#define XX(m, t) \
    case m: \
        ret->m_datas[name].alloc(sizeof(t)); \
        break
    for (int i = 0; i < len; i++)
    {
        std::string name = std::string(fields[i].name, fields[i].name_length);
        ret->m_fields.push_back(name);

        switch (fields[i].type)
        {
            XX(MYSQL_TYPE_TINY, int8_t);
            XX(MYSQL_TYPE_SHORT, int16_t);
            XX(MYSQL_TYPE_LONG, int32_t);
            XX(MYSQL_TYPE_LONGLONG, int64_t);
            XX(MYSQL_TYPE_FLOAT, float);
            XX(MYSQL_TYPE_DOUBLE, double);
            XX(MYSQL_TYPE_TIMESTAMP, MYSQL_TIME);
            XX(MYSQL_TYPE_DATETIME, MYSQL_TIME);
            XX(MYSQL_TYPE_DATE, MYSQL_TIME);
            XX(MYSQL_TYPE_TIME, MYSQL_TIME);
            default:
                ret->m_datas[name].alloc(fields[i].length);
                break;
        }

        ret->m_datas[name].buffer_type = fields[i].type;

        ret->m_binds[i].length = &ret->m_datas[name].length;
        ret->m_binds[i].is_null = &ret->m_datas[name].is_null;
        ret->m_binds[i].buffer = ret->m_datas[name].buffer;
        ret->m_binds[i].error = &ret->m_datas[name].error;
        ret->m_binds[i].buffer_length = ret->m_datas[name].buffer_length;
        ret->m_binds[i].buffer_type = ret->m_datas[name].buffer_type;
    }
#undef XX

    if (mysql_stmt_bind_result(stmt->get(), &ret->m_binds[0]))
    {
        std::cout << "mysql_stmt_bind_result error, errno="
            << stmt->getErrno() << ", errstr=" << stmt->getErrstr() << std::endl;
        return nullptr;
    }

    if (mysql_stmt_execute(stmt->get()))
    {
        std::cout << "mysql_stmt_execute error, errno="
            << stmt->getErrno() << ", errstr=" << stmt->getErrstr() << std::endl;
        return nullptr;
    }

    if (mysql_stmt_store_result(stmt->get()))
    {
        std::cout << "mysql_stmt_store_result error, errno="
            << stmt->getErrno() << ", errstr=" << stmt->getErrstr() << std::endl;
        return nullptr;
    }

    return ret;
}

bool MySQLStmtRes::next()
{
    return !mysql_stmt_fetch(m_stmt->get());
}

uint64_t MySQLStmtRes::getRows()
{
    return mysql_stmt_num_rows(m_stmt->get());
}


// static
MySQLStmt::ptr MySQLStmt::create(MySQL::ptr sql, const std::string& stmt)
{
    auto st = mysql_stmt_init(sql->get().get());
    if (!st)
        return nullptr;
    if (mysql_stmt_prepare(st, stmt.c_str(), stmt.size()))
    {
        std::cout << "stmt=" << stmt << ", errno=" << mysql_stmt_errno(st)
            << ", errstr=" << mysql_stmt_error(st) << std::endl;
        mysql_stmt_close(st);
        return nullptr;
    }
    uint64_t count = mysql_stmt_param_count(st);
    MySQLStmt::ptr ret(new MySQLStmt(sql, st));
    ret->m_binds.resize(count);
    memset(&ret->m_binds[0], 0, sizeof(ret->m_binds[0]) * count);
    return ret;
}

bool MySQLStmt::execute()
{
    mysql_stmt_bind_param(m_stmt, &m_binds[0]);
    if (mysql_stmt_execute(m_stmt))
    {
        std::cout << "mysql_stmt_execute error, errno="
            << getErrno() << ", errstr=" << getErrstr() << std::endl;
        return false;
    }
    return true;
}

MySQLStmtRes::ptr MySQLStmt::query()
{
    mysql_stmt_bind_param(m_stmt, &m_binds[0]);
    return MySQLStmtRes::create(shared_from_this());
}


MySQL::MySQL(const std::string& host, int port, const std::string& user,
           const std::string& passwd, const std::string& dbname, uint32_t poolSize)
    :m_host(host)
    ,m_port(port)
    ,m_user(user)
    ,m_passwd(passwd)
    ,m_dbname(dbname)
    ,m_poolSize(poolSize)
{
}

bool MySQL::connect()
{
    static thread_local MySQLThreadInit s_thread_init;

    if (m_mysql && !m_hasError)
        return true;

    MYSQL* mysql = ::mysql_init(nullptr);
    if (mysql == nullptr)
    {
        std::cout << "mysql_init error" << std::endl;
        m_hasError = true;
        return false;
    }

    int auto_reconnect = 0;
    mysql_options(mysql, MYSQL_OPT_RECONNECT, &auto_reconnect);
    mysql_options(mysql, MYSQL_SET_CHARSET_NAME, "utf8mb4"); // ��utf8mb4��Ϊ�˼���unicode
    if (mysql_real_connect(mysql, m_host.c_str(), m_user.c_str(),
            m_passwd.c_str(), m_dbname.c_str(), m_port, NULL, 0) == nullptr)
    {
        std::cout << "mysql_real_connect(" << m_host
                  << ", " << m_port << ", " << m_dbname
                  << ") error: " << mysql_error(mysql) << std::endl;
        mysql_close(mysql);
        m_hasError = true;
        return false;
    }

    m_hasError = false;
    m_mysql.reset(mysql, mysql_close);
    return true;
}

bool MySQL::ping()
{
    if (!m_mysql)
        return false;

    if (mysql_ping(m_mysql.get()))
    {
        m_hasError = true;
        return false;
    }

    m_hasError = false;
    return true;
}

bool MySQL::use(const std::string& dbname)
{
    if (!m_mysql)
        return false;
    if (m_dbname == dbname)
        return true;

    if (!mysql_select_db(m_mysql.get(), dbname.c_str()))
    {
        m_dbname = dbname;
        m_hasError = false;
        return true;
    }
    else
    {
        m_dbname = "";
        m_hasError = true;
        return false;
    }
}

bool MySQL::execute(const char* format, ...)
{
    std::string cmd;
    {
        va_list ap;
        va_start(ap, format);
        char* buf = nullptr;
        int len = vasprintf(&buf, format, ap);
        if (len != -1)
        {
            cmd.append(buf, len);
            free(buf);
        }
        va_end(ap);
    }

    if (!m_mysql)
    {
        std::cout << "m_mysql is NULL" << std::endl;
        m_hasError = true;
        return false;
    }

    int ret = ::mysql_real_query(m_mysql.get(), &cmd[0], cmd.size());
    if (ret)
    {
        std::cout << "sql: " << cmd << ", error: " << getErrstr() << std::endl;
        m_hasError = true;
    }
    m_hasError = false;
    return !ret;
}

bool MySQL::execute(const std::string& cmd)
{
    if (!m_mysql)
    {
        std::cout << "m_mysql is NULL" << std::endl;
        m_hasError = true;
        return false;
    }

    int ret = ::mysql_real_query(m_mysql.get(), &cmd[0], cmd.size());
    if (ret)
    {
        std::cout << "mysql_real_query(" << cmd << ") error:" << getErrstr() << std::endl;
        m_hasError = true;
    }

    m_hasError = false;
    return !ret;
}

MySQLRes::ptr MySQL::query(const char* format, ...)
{
    if (!m_mysql)
    {
        std::cout << "m_mysql is NULL" << std::endl;
        m_hasError = true;
        return nullptr;
    }

    std::string cmd;
    {
        va_list ap;
        va_start(ap, format);
        char* buf = nullptr;
        int len = vasprintf(&buf, format, ap);
        if (len != -1)
        {
            cmd.append(buf, len);
            free(buf);
        }
        va_end(ap);
    }

    if (::mysql_real_query(m_mysql.get(), &cmd[0], cmd.size()))
    {
        std::cout << "mysql_real_query(" << cmd << ") error:" << getErrstr() << std::endl;
        m_hasError = true;
        return nullptr;
    }
    MYSQL_RES* res = mysql_store_result(m_mysql.get());
    if (res == nullptr)
    {
        std::cout << "mysql_store_result(" << cmd << ") error:" << getErrstr() << std::endl;
        m_hasError = true;
        return nullptr;
    }

    m_hasError = false;
    MySQLRes::ptr ret(new MySQLRes(res));
    return ret;
}

MySQLRes::ptr MySQL::query(const std::string& cmd)
{
    if (!m_mysql)
    {
        std::cout << "m_mysql is NULL" << std::endl;
        m_hasError = true;
        return nullptr;
    }

    if (::mysql_real_query(m_mysql.get(), &cmd[0], cmd.size()))
    {
        std::cout << "mysql_query(" << cmd << ") error:" << getErrstr() << std::endl;
        m_hasError = true;
        return nullptr;
    }
    MYSQL_RES* res = mysql_store_result(m_mysql.get());
    if (res == nullptr)
    {
        std::cout << "mysql_store_result(" << cmd << ") error:" << getErrstr() << std::endl;
        m_hasError = true;
        return nullptr;
    }

    m_hasError = false;
    MySQLRes::ptr ret(new MySQLRes(res));
    return ret;
}

std::shared_ptr<MySQLTransaction> MySQL::openTransaction(bool auto_commit)
{
    return MySQLTransaction::create(shared_from_this(), auto_commit);
}

MySQLStmt::ptr MySQL::openPrepare(const std::string& cmd)
{
    return MySQLStmt::create(shared_from_this(), cmd);
}


// static
MySQLTransaction::ptr MySQLTransaction::create(MySQL::ptr mysql, bool auto_commit)
{
    MySQLTransaction::ptr ret(new MySQLTransaction(mysql, auto_commit));
    return ret->begin() ? ret : nullptr;
}

bool MySQLTransaction::execute(const char* format, ...)
{
    std::string cmd;
    {
        va_list ap;
        va_start(ap, format);
        char* buf = nullptr;
        int len = vasprintf(&buf, format, ap);
        if (len != -1)
        {
            cmd.append(buf, len);
            free(buf);
        }
        va_end(ap);
    }

    if (m_isFinished)
    {
        std::cout << "transaction is finished, sql: " << cmd << std::endl;
        return false;
    }

    bool ret = m_mysql->execute(cmd);
    if (!ret)
        m_hasError = true;
    return ret;
}

bool MySQLTransaction::execute(const std::string& cmd)
{
    if (m_isFinished)
    {
        std::cout << "transaction is finished, sql: " << cmd << std::endl;
        return false;
    }

    bool ret = m_mysql->execute(cmd);
    if (!ret)
        m_hasError = true;
    return ret;
}


MySQLManager::MySQLManager()
{
    mysql_library_init(0, nullptr, nullptr);
}

MySQLManager::~MySQLManager()
{
    mysql_library_end();
    for (auto& i : m_connections)
    {
        for (auto& n : i.second)
            delete n;
    }
}

void MySQLManager::add(const std::string& name, const std::string& host, int port,
                       const std::string& user, const std::string& passwd,
                       const std::string& dbname, uint32_t poolSize)
{
    MutexType::Lock lock(m_mutex);
    MySqlConf conf;
    conf.host = host;
    conf.port = port;
    conf.user = user;
    conf.passwd = passwd;
    conf.dbname = dbname;
    conf.poolSize = poolSize;
    m_sqlDefines[name] = conf;
}

MySQL::ptr MySQLManager::get(const std::string& name)
{
    MutexType::Lock lock(m_mutex);
    auto it = m_connections.find(name);
    if (it != m_connections.end())
    {
        if (!it->second.empty())
        {
            MySQL* ret = it->second.front();
            it->second.pop_front();
            lock.unlock();
            if (!ret->needToCheck())
            {
                return MySQL::ptr(ret, std::bind(&MySQLManager::freeMySQL, this,
                           name, std::placeholders::_1));
            }
            if (ret->ping())
            {
                return MySQL::ptr(ret, std::bind(&MySQLManager::freeMySQL, this,
                           name, std::placeholders::_1));
            }
            else if (ret->connect())
            {
                ret->setLastUsedTime(time(0));
                return MySQL::ptr(ret, std::bind(&MySQLManager::freeMySQL, this,
                           name, std::placeholders::_1));
            }
            else
            {
                std::cout << "reconnect sql(name: " << name << ") fail" << std::endl;
                return nullptr;
            }
        }
    }
    auto n = m_sqlDefines.find(name);
    if (n == m_sqlDefines.end())
    {
        std::cout << "get sql(name: " << name << ") fail" << std::endl;
        return nullptr;
    }
    lock.unlock();

    MySQL* ret = new MySQL(n->second.host, n->second.port, n->second.user,
                         n->second.passwd, n->second.dbname, n->second.poolSize);
    if (ret->connect())
    {
        ret->setLastUsedTime(time(0));
        return MySQL::ptr(ret, std::bind(&MySQLManager::freeMySQL, this,
                   name, std::placeholders::_1));
    }
    else
    {
        delete ret;
        return nullptr;
    }
}

bool MySQLManager::execute(const std::string& name, const char* format, ...)
{
    std::string cmd;
    {
        va_list ap;
        va_start(ap, format);
        char* buf = nullptr;
        int len = vasprintf(&buf, format, ap);
        if (len != -1)
        {
            cmd.append(buf, len);
            free(buf);
        }
        va_end(ap);
    }

    MySQL::ptr connection = get(name);
    if (!connection)
    {
        std::cout << "MySQLManager::execute, get(" << name << ") fail, sql: " << cmd << std::endl;
        return false;
    }
    return connection->execute(cmd);
}

bool MySQLManager::execute(const std::string& name, const std::string& cmd)
{
    MySQL::ptr connection = get(name);
    if (!connection)
    {
        std::cout << "MySQLManager::execute, get(" << name << ") fail, sql: " << cmd << std::endl;
        return false;
    }
    return connection->execute(cmd);
}

MySQLRes::ptr MySQLManager::query(const std::string& name, const char* format, ...)
{
    std::string cmd;
    {
        va_list ap;
        va_start(ap, format);
        char* buf = nullptr;
        int len = vasprintf(&buf, format, ap);
        if (len != -1)
        {
            cmd.append(buf, len);
            free(buf);
        }
        va_end(ap);
    }

    MySQL::ptr connection = get(name);
    if (!connection)
    {
        std::cout << "MySQLManager::query, get(" << name << ") fail, sql: " << cmd << std::endl;
        return nullptr;
    }
    return connection->query(cmd);
}

MySQLRes::ptr MySQLManager::query(const std::string& name, const std::string& cmd)
{
    MySQL::ptr connection = get(name);
    if (!connection)
    {
        std::cout << "MySQLManager::query, get(" << name << ") fail, sql: " << cmd << std::endl;
        return nullptr;
    }
    return connection->query(cmd);
}

MySQLTransaction::ptr MySQLManager::openTransaction(const std::string& name,
                                        bool auto_commit)
{
    auto connection = get(name);
    if (!connection)
    {
        std::cout << "MySQLManager::openTransaction: get(" << name << ") fail" << std::endl;
        return nullptr;
    }
    return connection->openTransaction(auto_commit);
}

MySQLStmt::ptr MySQLManager::openPrepare(const std::string& name, const std::string& cmd)
{
    auto connection = get(name);
    if (!connection)
    {
        std::cout << "MySQLManager::openPrepare: get(" << name << ") fail" << std::endl;
        return nullptr;
    }
    return connection->openPrepare(cmd);
}

void MySQLManager::checkConnection(int sec)
{
    time_t now = time(0);
    std::vector<MySQL*> connections;
    MutexType::Lock lock(m_mutex);
    for (auto& i : m_connections)
    {
        for (auto it = i.second.begin(); it != i.second.end();)
        {
            if ((int)(now - (*it)->getLastUsedTime()) >= sec)
            {
                connections.push_back(*it);
                i.second.erase(it++);
            }
            else
                it++;
        }
    }
    lock.unlock();
    for (auto& i : connections)
        delete i;
}

// private
void MySQLManager::freeMySQL(const std::string& name, MySQL* m)
{
    MutexType::Lock lock(m_mutex);
    if (m_connections[name].size() < (size_t)m->getPoolSize())
    {
        m_connections[name].push_back(m);
        return;
    }
    delete m;
}

}

namespace MySQLUtil
{

bool execute(const std::string& name, const char* format, ...)
{
    std::string cmd;
    {
        va_list ap;
        va_start(ap, format);
        char* buf = nullptr;
        int len = vasprintf(&buf, format, ap);
        if (len != -1)
        {
            cmd.append(buf, len);
            free(buf);
        }
        va_end(ap);
    }
    return mysql::MySQLMgr::GetInstance()->execute(name, cmd);
}

bool execute(const std::string& name, const std::string& cmd)
{
    return mysql::MySQLMgr::GetInstance()->execute(name, cmd);
}

mysql::MySQLRes::ptr query(const std::string& name, const char* format, ...)
{
    std::string cmd;
    {
        va_list ap;
        va_start(ap, format);
        char* buf = nullptr;
        int len = vasprintf(&buf, format, ap);
        if (len != -1)
        {
            cmd.append(buf, len);
            free(buf);
        }
        va_end(ap);
    }
    return mysql::MySQLMgr::GetInstance()->query(name, cmd.c_str());
}

mysql::MySQLRes::ptr query(const std::string& name, const std::string& cmd)
{
    return mysql::MySQLMgr::GetInstance()->query(name, cmd);
}

mysql::MySQLTransaction::ptr openTransaction(const std::string& name,
                                 bool auto_commit)
{
    return mysql::MySQLMgr::GetInstance()->openTransaction(name, auto_commit);
}

mysql::MySQLStmt::ptr openPrepare(const std::string& name, const std::string& cmd)
{
    return mysql::MySQLMgr::GetInstance()->openPrepare(name, cmd);
}

void checkConnection(int sec)
{
    mysql::MySQLMgr::GetInstance()->checkConnection(sec);
}

}
