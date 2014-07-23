/*
 FALCON - The Falcon Programming Language.
 FILE: shmem_darwin.h
 
 Semaphore timed wait for APPLE
 
 -------------------------------------------------------------------
 Author: Chris Misztur
 Begin: Tue, 22 Jul 2014
 
 -------------------------------------------------------------------
 (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 
 See LICENSE file for licensing details.
 */

#ifndef __OSX_SEM_TIMEDWAIT_H_
#define __OSX_SEM_TIMEDWAIT_H_


// Mac OS X does not have a working implementation of sem_init, sem_timedwait, ...
// use Mach semaphores instead
#include <mach/mach.h>
#include <mach/semaphore.h>
#include <mach/task.h>

// Mac OS X timedwait wrapper
//int sem_timedwait_mach(semaphore_t* sem, long timeout_ms);
int sem_timedwait_mach(sem_t *sem, const struct timespec *abs_timeout);

#endif