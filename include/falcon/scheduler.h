/*
   FALCON - The Falcon Programming Language.
   FILE: scheduler.h

   Timer for delayed activities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 13 Feb 2013 15:17:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SCHEDULER_H_
#define _FALCON_SCHEDULER_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/mt.h>

namespace Falcon
{

/** Timer for delayed activities.
 *
 * This is a generic class keeping track of activities that
 * are to be delayed and are scheduled to happen at a future time.
 *
 * It is used by the virtual machine and the processors to take
 *
 */
class FALCON_DYN_CLASS Scheduler: public Runnable
{
public:
   Scheduler( );
   virtual ~Scheduler( );

   class Activity;

   typedef void(t_callback)( void* data, Activity* activity );

   /** Schedules an activity to be performed after a certain time.
    * \param delay in milliseconds since current time after which the callback will be invoked.
    * \param cb the callback that will be invoked.
    * \param data The data that will be passed to the callback.
    * \param cancelable If true, cancelActivity can be invoked to cancel the pending activity.
    * \return A new cancleable activity, or 0 if the activity is not cancelable.
    *
    * The callback activity will be called by another thread at given timeout.
    *
    * If the activity is cancelable, an Activity instance (which is opaque at
    * caller level) is returned. The caller \b must then invoke one and just one
    * of cancelActivity() or completeActivity().
    * */
   Activity* addActivity( uint32 delay, t_callback &cb, void* data, bool cancelable = false );

   /** Schedules an activity to be performed at a given time in future.
    * \param ts Time in future in milliseconds sinche epoch.
    * \param cb the callback that will be invoked.
    * \param data The data that will be passed to the callback.
    * \param cancelable If true, cancelActivity can be invoked to cancel the pending activity.
    * \return A new cancleable activity, or 0 if the activity is not cancelable.
    *
    * The callback activity will be called by another thread at given timeout.
    *
    * \note If \b ts is not in future, the activity gets called as soon as possible
    * by the scheduler thread.
    *
    * If the activity is cancelable, an Activity instance (which is opaque at
    * caller level) is returned. The caller \b must then invoke one and just one
    * of cancelActivity() or completeActivity().
    * */
   Activity* addActivityAt( int64 ts, t_callback &cb, void* data, bool cancelable = false );

   /** Cancel a given pending activity.
    * \param activity The activity created by addActivity() to be disposed of.
    * \return true if the activity could be remvoed; false if it has been already invoked.
    *
    * If the method returns true, then it is granted that the activity
    * callback won't be invoked; the caller can safely dispose of the associated
    * data and assume that the callback wasn't called and won't be called in the future.
    *
    * If the method returns false, then the callback is either being called or has
    * been already called. Either way, the caller must assume that the data cannot
    * be disposed, and must prepare to dispose it in the callback.
    *
    * After this call, the caller must consider the activity object disposed and
    * invalid.
    *
    * \note The owner of the scheduled activity \b must invoke either cancelActivity
    * or completeActivity.
    */
   bool cancelActivity( Activity* activity );

   /** Declare an activity complete.
    * \param activity the activity now complete.
    *
    * This method can be called by the callback function of the activity
    * to declare that the activity is complete and cancelActivity won't be called.
    * The method can be called at any moment during the callback execution, but
    * it should not be invoked by other code.
    *
    * After this call the \b activity must be consider invalid. Referencing it
    * has undefined result.
    */
   void completeActivity( Activity* activity );

   /**
    * Stops the scheduler.
    *
    * Pending schedules are \b not respected.
    */
   void stop();

   virtual void* run();

private:
   SysThread* m_thread;
   bool m_terminated;
   Mutex m_mtx;
   Event m_evtActivity;

   class Private;
   Scheduler::Private* _p;
};

}

#endif

/* scheduler.h */
