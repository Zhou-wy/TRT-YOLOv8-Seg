#ifndef __NONCOPYABLE_H
#define __NONCOPYABLE_H

struct Noncopyable
{
    Noncopyable() = default;

    ~Noncopyable() = default;

    Noncopyable(const Noncopyable&) = delete;

    Noncopyable& operator=(const Noncopyable&) = delete;
};

#endif /*__NONCOPYABLE_H*/

