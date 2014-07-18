/*
 FALCON - The Falcon Programming Language.
 FILE: ipsem_darwin.cpp
 
 Inter-process semaphore -- DARWIN specific
 -------------------------------------------------------------------
 Author: Giancarlo Niccolai
 Begin: Mon, 11 Nov 2013 16:27:30 +0100
 
 -------------------------------------------------------------------
 (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)
 
 See LICENSE file for licensing details.
 */

#define SRC "modules/native/feathers/shmem/ipsem_darwin.cpp"

#include <falcon/autocstring.h>
#include <falcon/stderrors.h>

#include <fcntl.h>
#include <time.h>
#include <semaphore.h>

#include "ipsem.h"
#include "errors.h"

#include <mach/clock.h>
#include <mach/mach.h>

#include "shmem_darwin.h"

#define NAME_PREFIX "/FSHM-"

namespace Falcon
{
    
    class IPSem::Private
    {
    public:
        
        Private() {
            hasInit = 0;
            semaphore = SEM_FAILED;
            
        }
        ~Private() {}
        
        sem_t* semaphore;
        atomic_int hasInit;
        String semName;
    };
    
    IPSem::IPSem()
    {
        _p = new Private;
    }
    
    IPSem::IPSem( const String& name )
    {
        _p = new Private;
        init( name, e_om_create );
    }
    
    
    IPSem::IPSem( const IPSem& other )
    {
        _p = new Private;
        init( other._p->semName, e_om_open );
    }
    
    
    IPSem::~IPSem()
    {
        close();
        delete _p;
    }
    
    
    
    void IPSem::init( const String& name, IPSem::t_open_mode mode, bool bPublic )
    {
        if( ! atomicCAS(_p->hasInit, 0, 1) )
        {
            throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_ALREADY_INIT);
        }
        
        // add a standard prefix if necessary
        String sname;
        if( ! name.empty() && name.getCharAt(0) != '/' )
        {
            sname = String(NAME_PREFIX) + name;
        }
        else {
            sname = name;
        }
        
        _p->semName = sname;
        
        int omode, pmode;
        switch(mode)
        {
            case e_om_create: omode = O_CREAT | O_EXCL; break;
            case e_om_open: omode = O_CREAT; break;
            case e_om_open_existing: omode = 0; break;
        }
        
        pmode = bPublic ? 0777: 0700;
        
        AutoCString cname(sname);
        if( omode != 0 ) {
            _p->semaphore = sem_open( cname.c_str(), omode, pmode, 0 );
        }
        else {
            _p->semaphore = sem_open( cname.c_str(), 0 );
        }
        
        if( _p->semaphore == SEM_FAILED )
        {
            if( _p->semaphore == SEM_FAILED )
            {
                throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_INIT, .sysError((uint32) errno) );
            }
        }
    }
    
    
    void IPSem::close( bool bDelete )
    {
        if( atomicFetch(_p->hasInit ) == 0 )
        {
            throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_NOT_INIT );
        }
        
        // ignore multiple close
        if(_p->semaphore == SEM_FAILED)
        {
            return;
        }
        
        int res = sem_close(_p->semaphore);
        if( res != 0 )
        {
            throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                                     .sysError((uint32) errno)
                                     .extra("sem_close") );
        }
        _p->semaphore = SEM_FAILED;
        
        if( bDelete )
        {
            AutoCString cname(_p->semName);
            res = sem_unlink(cname.c_str());
            if( res != 0 )
            {
                throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                                         .sysError((uint32) errno)
                                         .extra("sem_unlink"));
            }
        }
    }
    
    
    void IPSem::post()
    {
        if( atomicFetch(_p->hasInit ) == 0 || _p->semaphore == SEM_FAILED )
        {
            throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_NOT_INIT );
        }
        
        int res =  sem_post(_p->semaphore);
        if( res != 0 )
        {
            throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_POST,
                                     .sysError((uint32) errno)
                                     .extra("sem_close") );
        }
    }
    
    
    bool IPSem::wait( int64 to )
    {
        if( atomicFetch(_p->hasInit ) == 0 || _p->semaphore == SEM_FAILED )
        {
            throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_NOT_INIT );
        }
        
        int res = 0;
        do
        {
            if( to < 0 )
            {
                res = sem_wait(_p->semaphore);
            }
            else if( to == 0 )
            {
                res = sem_trywait(_p->semaphore);
            }
            else
            {
                struct timespec ts_wait;
                struct timespec tv_now;
                
                // OS X does not have clock_gettime, use clock_get_time
                clock_serv_t cclock;
                mach_timespec_t mts;
                host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
                clock_get_time(cclock, &mts);
                mach_port_deallocate(mach_task_self(), cclock);
                tv_now.tv_sec = mts.tv_sec;
                tv_now.tv_nsec = mts.tv_nsec;
                
                int msec = (to % 1000);
                ts_wait.tv_nsec = tv_now.tv_nsec + (msec * 1000000);
                ts_wait.tv_sec = tv_now.tv_sec + (to/1000);
                if ( ts_wait.tv_nsec > 1000000000 )
                {
                    ts_wait.tv_nsec -= 1000000000;
                    ts_wait.tv_sec++;
                }
                
                //https://github.com/constcast/vermont/tree/master/src/osdep/osx
                //see also : https://github.com/attie/libxbee3/blob/master/xsys_darwin/sem_timedwait.c
                res = sem_timedwait_mach(_p->semaphore, &ts_wait);
            }
        }
        // repeat in case of signal interruption.
        while( res != 0 && errno == EINTR );
        
        if( res != 0 )
        {
            if( errno == ETIMEDOUT || errno == EAGAIN )
            {
                return false;
            }
            throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_WAIT, .sysError((uint32) errno) );
        }
        
        return true;
    }
    
}

/* end of ipsem_darwin.cpp */
