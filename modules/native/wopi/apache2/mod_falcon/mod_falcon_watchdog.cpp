/*
   FALCON - The Falcon Programming Language.
   FILE: mod_falcon_watchdog.h

   Falcon module for Apache 2

   Parallel watchdog to stop a runaway VM.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Mar 2012 16:59:57 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "mod_falcon_watchdog.h"
#include "apache_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <apache2/httpd.h>

#include <falcon/stream.h>

FALCON_WATCHDOG_DATA falconWatchdogData; 


//===============================================================
// Public interface 
//===============================================================

FALCON_WATCHDOG_TOKEN* watchdog_push_vm( 
      request_rec* request, ApacheVMachine* vm, 
      int timeout, ApacheReply* reply )
{
   // get current time.
   apr_time_t now = apr_time_now();
   apr_pool_t* pool = request->pool;
   
   FALCON_WATCHDOG_TOKEN* token = (FALCON_WATCHDOG_TOKEN*) apr_palloc( pool, sizeof(FALCON_WATCHDOG_TOKEN) );
   token->hasCompleted = 0;
   token->wasStopped = 0;
   token->stopTimeout = timeout/4;
   token->expireTime = now + (timeout * 1000000);   
   token->vm = vm;
   token->request = request;
   token->reply = reply;

   apr_thread_mutex_lock( falconWatchdogData.mutex );
   if( falconWatchdogData.incomingVM == 0 )
   {
      falconWatchdogData.lastIncomingVM = token;
   }   
   token->next = falconWatchdogData.incomingVM;
   falconWatchdogData.incomingVM = token;
   
   apr_thread_cond_signal( falconWatchdogData.cond );
   apr_thread_mutex_unlock( falconWatchdogData.mutex );
}


void watchdog_vm_is_done( FALCON_WATCHDOG_TOKEN* token )
{   
   apr_thread_mutex_lock( falconWatchdogData.mutex );
   token->hasCompleted = 1;
   falconWatchdogData.vmHasDone = 1; 
   token->wasStopped = 0;
   apr_thread_cond_signal( falconWatchdogData.cond );
   apr_thread_mutex_unlock( falconWatchdogData.mutex );
}

//===============================================================
// Private part
//===============================================================

static void watchdog_abort( FALCON_WATCHDOG_TOKEN* failing )
{
   // we're condamned.
   apr_table_add(failing->request->headers_out, "Content-type", "text/plain" );
   failing->request->status = 408;
   failing->request->status_line = "Request Timeout";
   ApacheOutput* aout = failing->reply->aout();
   aout->write("\nFalcon::WOPI (aout) - Connection forcefully broken due to timeout\n");
   aout->close();
   
   fprintf( stderr, "Timed out while trying to get the VM back online after %d secs:\n"
      "URI: %s\nFile: %s\n",
      (int) failing->stopTimeout,
      failing->request->unparsed_uri,
      failing->request->canonical_filename );
   
   sleep(2);
   exit(1);
}
// Checks if there is some VM to be purged.
static apr_time_t purgeOldVM()
{
   apr_time_t expireMin = 0;
   apr_time_t now = apr_time_now();
   
   FALCON_WATCHDOG_TOKEN *old = NULL;
   FALCON_WATCHDOG_TOKEN *token = falconWatchdogData.activeVM;
   
   while ( token != NULL )
   {
      // is this vm done?
      //apr_thread_mutex_lock(moduleWatchdogData.mutex);
      int bDone = token->hasCompleted;
      //apr_thread_mutex_unlock(moduleWatchdogData.mutex);
      
      if( bDone )
      {
         if( old != 0 ) {
            old->next = token->next;
            // we can free this token.
            // The watchdog token was allocated in the pool of the request,
            // we don't need to free it (will be destroyed at request termination).
         }
         else {
            falconWatchdogData.activeVM = token->next;
         }
      }
      else 
      {
         // shall this VM be killed?
         if( token->expireTime <= now )
         {
            // bad news for the vm.
            if( token->wasStopped )
            {
               // and bad news for us as well. We cannot recover.
               watchdog_abort( token );
            }
            else 
            {
               // try to stop the vm.
               token->vm->terminate();
               token->wasStopped = 1;
               token->expireTime = now + (token->stopTimeout * 1000000);
            }
         }
         
         if ( expireMin == 0 || token->expireTime < expireMin )
         {
            expireMin = token->expireTime;
         }
      }
      
      old = token;
      token = token->next;      
   }
   
   return expireMin == 0 ? 0 : expireMin - now;
}

void* APR_THREAD_FUNC watchdog(apr_thread_t *thd, void *data)
{
   FALCON_WATCHDOG_DATA* wdata = (FALCON_WATCHDOG_DATA*) data;
   falconWatchdogData.activeVM = NULL;
   apr_time_t maxWait = 0;
   
   fprintf(stderr, "Falcon module -- Watchdog start\n" );
   
   // While we're active, we want the lock for us.
   while( true )
   {   
      apr_thread_mutex_lock(wdata->mutex);
      
      int vmHasDone = wdata->vmHasDone;
      while( wdata->incomingVM == NULL && ! vmHasDone ) 
      {
         if( wdata->terminate )
         {
            // we're done   
            apr_thread_mutex_unlock(wdata->mutex);
            goto terminate_thread;
         }
         
         if( maxWait <= 0 )
         {
            apr_thread_cond_wait(wdata->cond, wdata->mutex);
         }
         else {
            apr_thread_cond_timedwait( wdata->cond, wdata->mutex, maxWait );
            vmHasDone = 1; // force to check the status of our our vms nevertheless
         }
      }     
      
      wdata->vmHasDone = 0;
      if( wdata->incomingVM != 0 )
      {
         wdata->lastIncomingVM = wdata->activeVM;
         wdata->activeVM = wdata->incomingVM;
         wdata->incomingVM = 0;
         apr_thread_mutex_unlock(wdata->mutex);
      }
      else {      
         apr_thread_mutex_unlock(wdata->mutex);
      }
      maxWait = purgeOldVM();      
   }
   
terminate_thread:   
   apr_thread_exit(thd, APR_SUCCESS);   
   return NULL;
}


