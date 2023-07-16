/**
 * brief: 单例类封装
 */
#ifndef __SINGLETON_H
#define __SINGLETON_H

template<typename T>
struct Singleton
{
    static T* GetInstance()
    {
        static T m;
        return &m;
    }
};

#endif /*__SINGLETON_H*/
