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

#ifndef MOD_FALCON_WATCHDOG_H
#define MOD_FALCON_WATCHDOG_H

#include "http_request.h"
#include <apr_general.h>
#include <apr_thread_proc.h>
#include <apr_thread_cond.h>

#include "mod_falcon_vm.h"
#include "apache_reply.h"

typedef struct tag_FALCON_WATCHDOG_TOKEN
{
   ApacheVMachine* vm;
   int hasCompleted;
   int wasStopped;
   
   apr_time_t stopTimeout;
   apr_time_t expireTime;
   request_rec* request;
   ApacheReply* reply;
   struct tag_FALCON_WATCHDOG_TOKEN* next;
}
FALCON_WATCHDOG_TOKEN;


typedef struct 
{
    apr_thread_t *watchdogThread;
   /* condition variable should be used with a mutex variable */
    apr_thread_mutex_t *mutex;
    apr_thread_cond_t  *cond;
   
    apr_pool_t* threadPool;

    /* shared context depends on application */    
    int terminate;    
    // This is guarded by the mutex
    FALCON_WATCHDOG_TOKEN* incomingVM;
    FALCON_WATCHDOG_TOKEN* lastIncomingVM;
    int vmHasDone;
    
    // This isn't, it's private.
    FALCON_WATCHDOG_TOKEN* activeVM;
} FALCON_WATCHDOG_DATA;


void* APR_THREAD_FUNC watchdog(apr_thread_t *thd, void *data);
FALCON_WATCHDOG_TOKEN* watchdog_push_vm( request_rec* pool, ApacheVMachine* vm, int timeout, ApacheReply* reply );
void watchdog_vm_is_done(FALCON_WATCHDOG_TOKEN* token);

extern FALCON_WATCHDOG_DATA falconWatchdogData; 

#endif
