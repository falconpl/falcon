/*
   FALCON - The Falcon Programming Language.
   FILE: ipsem_win.cpp

   Inter-process semaphore -- Windows specific
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Nov 2013 16:27:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/shmem/ipsem_win.cpp"

#include <windows.h>

#include <falcon/autowstring.h>
#include <falcon/stderrors.h>

#include "ipsem.h"
#include "errors.h"

namespace Falcon
{

class IPSem::Private
{
public:

   Private() {
      hasInit = 0;
      semaphore = INVALID_HANDLE_VALUE;

   }
   ~Private() {}

   HANDLE semaphore;
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



void IPSem::init( const String& name, IPSem::t_open_mode mode, bool pMode )
{
   if( ! atomicCAS(_p->hasInit, 0, 1) )
   {
      throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_ALREADY_INIT);
   }

   // add a standard prefix if necessary
   String sname;

   // create a semaphore in local space, by default
   if( name.find('\\') == String::npos )
   {
      if ( pMode ) {
         sname = "Global\\" + name;
      }
      else {
         sname = "Local\\" + name;
      }
   }
   else {
      sname = name;
   }

   _p->semName = sname;
   AutoWString cname(sname);

   switch( mode )
   {
   case e_om_create: 
      _p->semaphore = CreateSemaphoreW( NULL, 0, 10000, cname.w_str() );
      break;

   case e_om_open:
      _p->semaphore = CreateSemaphoreW( NULL, 0, 10000, cname.w_str() );
      if( _p->semaphore == INVALID_HANDLE_VALUE && GetLastError() == ERROR_ALREADY_EXISTS )
      {
         _p->semaphore = OpenSemaphoreW( SYNCHRONIZE | SEMAPHORE_MODIFY_STATE, FALSE, cname.w_str() );
      }
      break;

   case e_om_open_existing:      
      _p->semaphore = OpenSemaphoreW( SYNCHRONIZE | SEMAPHORE_MODIFY_STATE, FALSE, cname.w_str() );
      break;
   }

   if( _p->semaphore == INVALID_HANDLE_VALUE )
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_INIT, 
            .sysError((uint32) GetLastError()) );
   }
}


void IPSem::close( bool )
{
   if( atomicFetch(_p->hasInit ) == 0 )
   {
      throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_NOT_INIT );
   }

   // ignore multiple close
   if(_p->semaphore == INVALID_HANDLE_VALUE)
   {
      return;
   }

   if( ! CloseHandle(_p->semaphore) )
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
               .sysError((uint32) GetLastError())
               .extra("CloseHandle") );
   }
   _p->semaphore = INVALID_HANDLE_VALUE;
}


void IPSem::post()
{
   if( atomicFetch(_p->hasInit ) == 0 || _p->semaphore == INVALID_HANDLE_VALUE )
   {
      throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_NOT_INIT );
   }

   LONG lPrevCount = 0;
   BOOL res = ReleaseSemaphore(_p->semaphore, 1, &lPrevCount );
   if( ! res )
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_POST,
         .sysError((uint32) GetLastError())
               .extra("ReleaseSemaphore") );
   }
}


bool IPSem::wait( int64 to )
{
   if( atomicFetch(_p->hasInit ) == 0 || _p->semaphore == INVALID_HANDLE_VALUE )
   {
      throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_NOT_INIT );
   }

   DWORD res = WaitForSingleObject( _p->semaphore, static_cast<DWORD>(to < 0 ? INFINITE : to) );
   if( res == WAIT_TIMEOUT )
   {
      return false;
   }
   else if( res == WAIT_OBJECT_0 )
   {
      return true;
   }
   else {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_WAIT, .sysError((uint32) GetLastError()) );
   }

   return true;
}

}

/* end of ipsem_win.cpp */
