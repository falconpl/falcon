/*
   FALCON - The Falcon Programming Language.
   FILE: concurrencyguard.h

   Guard against unguarded concurrent access to a shared object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Apr 2013 16:29:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_CONCURRENCYGUARD_H
#define FALCON_CONCURRENCYGUARD_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/atomic.h>

namespace Falcon {

class VMContext;

/** Guard against unauthorized concurrent access to objects.
 *
 * This class is used to prevent two agents in the virtual machine to
 * perform unauthorized access to the same object without actually locking it.
 *
 * Using a low-level lock on an object that can be accessed concurrently is
 * not always the right way to handle concurrent access to objects. When
 * an unguarded access is almost surely a fault of the high-level program
 * logic, locking the access to the object can prevent an operating-system
 * level crash, but won't fix the logic error.
 *
 * Actually, by preventing the application to crash, a lock-based concurrency
 * policy might hide high-level logic problems, that could be more harmful
 * than the crashing of the application itself.
 *
 * For instance, Falcon language arrays are guarded against concurrent modify
 * through this mean. Suppose that a for/in loop is scanning the array elements,
 * and an agent removes some element while the second-last element is processed.
 * Even if, on the next loop, the for/in loop could exit without crashing the
 * host O/S level application, the for-last block of the for loop would be
 * erroneously skipped, causing possible data corruption in output.
 *
 * This guard system will cause the VM process to cleanly terminate in case an unauthorized
 * concurrent access is detected. This will rescue the host application from having
 * a S/O level crash, preventing the Falcon program to carry on potentially dangerous
 * activities.
 *
 * On the host application side, the correct action is that of detecting the process
 * error termination (which could happen for other reasons as well), and then fix
 * the problem by having the agents coordinating access to the shared object at
 * high-level, i.e. using a script level @a Semaphore to perform potentially harmful
 * changes.
 *
 * This class exposes two methods that take a VMContext as a parameter: read() and write().
 * A VMContext 'ctx' is said to enter the read guard if the ConcurrencyGuard::read(ctx) method is
 * invoked; similarly goes for entering the write guard.
 *
 * When no context is in write guard, any number of contexts can enter the read guard.
 * If more than one context tries to enter the write guard, or if any context tries to
 * enter the read guard while there is one context in the write guard, a ConcurrencyError
 * is thrown, and the process is terminated.
 *
 * Process termination requires all the contexts to be cleanly terminated prior returning
 * the control to the host application, so the process termination is not immediate, but
 * it will be performed as soon as possible. However, it is granted that the progress on the
 * offending context is immediately stopped.
 *
 * Also, any further context trying to enter the guard in read or write mode after an offense is
 * detected is granted to be immediately terminated with a ConcurrencyError throw.
 *
 * \note Two helper classes (Reder & Writer) can be used to wrap a protected section of code
 * in a read or write guard, automatically releasing the guard as the scope is exited.
  */
class FALCON_DYN_CLASS ConcurrencyGuard
{
public:
   static const int32 toeknWrite = 0;
   static const int32 toeknRead = 1;

   typedef int32 Token;

   ConcurrencyGuard();

   ~ConcurrencyGuard() {}

   /** Tries to enter the write guard.
    * \param ctx The context trying to enter the guard.
    *
    * If the guard can be entered, the method returns a token
    * that can be given back to the release() method.
    *
    * If not, the process owning the context is terminated,
    * and a ConcurrencyError is thrown.
    *
    * \note Write guards are not reentrant. If an agent tries
    * to acquire the write guard on an object more than once,
    * this will cause an offense detection.
    */
   Token write( VMContext* ctx );

   /** Tries to enter the write guard.
    * \param ctx The context trying to enter the guard.
    *
    * If the guard can be entered, the method returns a token
    * that can be given back to the release() method.
    *
    * If not, the process owning the context is terminated,
    * and a ConcurrencyError is thrown.
    *
    * \note Read guards are reentrant. It is possible for an
    * agent to acquire read guards on the same object multiple
    * times, provided it invokes release() an equal amount of
    * times.
    */
   Token read( VMContext* ctx );
   
   /** releases a previously acquired guard.
    *
    */
   inline void release( int32 token ) {
      if( token == 0 ) { releaseWrite(); }
      else { releaseRead(); }
   }

   void releaseWrite();
   void releaseRead();

   /** Helper class to safely guard a code section.
    *
    * Example usage:
    * \code
    * void* Guarded::fetchData( VMContext* ctx, int i )
    * {
    *    ConcurrencyGuard::Reader rguard(ctx, this->guard );
    *    // do stuff...
    *    return this->elementAt(i)
    * }
    *
    * \endcode
    *
    */
   class Reader {
   public:
      Reader( VMContext* ctx, ConcurrencyGuard& tg ):
         m_tg(tg)
      {
         tg.read( ctx );
      }

      ~Reader() {
         m_tg.releaseRead();
      }
   private:
      ConcurrencyGuard& m_tg;
   };


   /** Helper class to safely guard a code section.
    *
    * Example usage:
    * \code
    * void Guarded::setData( VMContext* ctx, int i, void* data )
    * {
    *    ConcurrencyGuard::Writer wguard(ctx, this->guard );
    *    // do stuff...
    *    this->setElementAt(i, data);
    * }
    *
    * \endcode
    */
   class Writer {
      public:
         Writer( VMContext* ctx, ConcurrencyGuard& tg ):
            m_tg(tg)
         {
            tg.write( ctx );
         }

         ~Writer() {
            m_tg.releaseWrite();
         }
      private:
         ConcurrencyGuard& m_tg;
      };

private:
   atomic_int m_readCount;
   atomic_int m_writeCount;
};

}

#endif

/* end of concurrencyguard.h */
