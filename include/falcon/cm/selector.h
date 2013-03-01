/*
   FALCON - The Falcon Programming Language.
   FILE: selector.h

   Falcon core module -- Semaphore shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Feb 2013 22:34:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_SELECTOR_H
#define FALCON_CORE_SELECTOR_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classshared.h>

#include <falcon/method.h>

namespace Falcon {
namespace Ext {

/*#
 @class Selector
 @brief Selects ready streams.
 @param fair True to create an acquirable (fair) selector.

 */
class FALCON_DYN_CLASS ClassSelector: public ClassShared
{
public:
   ClassSelector();
   virtual ~ClassSelector();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;

private:

   /*#
     @method add Selector
     @brief Adds a stream to the selector with a given mode
     @param stream The stream to be added
     @optparam mode Mode of read-write all.

    */
   FALCON_DECLARE_METHOD( add, "stream:Stream, mode:[N]" );

   /*#
     @method update Selector
     @brief Changes the stream selection mode.
     @param stream The stream to be updated
     @param mode Mode of read-write.

    */
   FALCON_DECLARE_METHOD( update, "stream:Stream, mode:N" );

   /*#
     @method addRead Selector
     @brief Adds a stream to the selector for read operations
     @param stream The stream to be added for reading.
    */
   FALCON_DECLARE_METHOD( addRead, "stream:Stream" );

   /*#
     @method addWrite Selector
     @brief Adds a stream to the selector for write operations
     @param stream The stream to be added for writing.
    */
   FALCON_DECLARE_METHOD( addWrite, "stream:Stream" );

   /*#
     @method addWrite Selector
     @brief Adds a stream to the selector for errors and out-of-band operations
     @param stream The stream to be added for errors or out of band data.
    */
   FALCON_DECLARE_METHOD( addErr, "stream:Stream" );



   /*#
     @method getRead Selector
     @brief Gets the next ready-for-read stream.
     @return Next ready stream.

     If this selector is created in fair mode, the get operation will
     exit the critical section acquired at wait success.

     If the method is not in fair mode, and there are multiple waiters
     for this selector, the get method might return nil, as the ready
     streams might get dequeued by other agents before this method is
     complete.

     Also, the method will return nil if the wait successful for other
     operations.
    */
   FALCON_DECLARE_METHOD( getRead, "" );

   /*#
     @method getWrite Selector
     @brief Gets the next ready-for-write stream.
     @return Next ready stream.

     If this selector is created in fair mode, the get operation will
     exit the critical section acquired at wait success.

     If the method is not in fair mode, and there are multiple waiters
     for this selector, the get method might return nil, as the ready
     streams might get dequeued by other agents before this method is
     complete.

     Also, the method will return nil if the wait successful for other
     operations.
    */
   FALCON_DECLARE_METHOD( getWrite, "" );

   /*#
     @method getErr Selector
     @brief Gets the next out-of-band signaled stream.
     @return Next ready stream.

     If this selector is created in fair mode, the get operation will
     exit the critical section acquired at wait success.

     If the method is not in fair mode, and there are multiple waiters
     for this selector, the get method might return nil, as the ready
     streams might get dequeued by other agents before this method is
     complete.

     Also, the method will return nil if the wait successful for other
     operations.
    */
   FALCON_DECLARE_METHOD( getErr, "" );


   /*#
     @method get Selector
     @brief Gets the next ready stream.
     @return Next ready stream.

     This method returns the first ready stream, peeked in order in:
     - Out of band data signaled queue.
     - Ready for read signaled queue.
     - Ready for write signaled queue.

     The method doesn't indicate what the operation the stream is ready
     for; so if this method is used, other means need to be available
     to know how to use the returned stream.

     If this selector is created in fair mode, the get operation will
     exit the critical section acquired at wait success.

     If the method is not in fair mode, and there are multiple waiters
     for this selector, the get method might return nil, as the ready
     streams might get dequeued by other agents before this method is
     complete.

     Also, the method will return nil if the wait successful for other
     operations.
    */
   FALCON_DECLARE_METHOD( get, "" );


   /*#
     @method tryWait Selector
     @brief Check if some of the streams are ready.

     The check eventually resets the semaphore if it's currently signaled.
    */
   FALCON_DECLARE_METHOD( tryWait, "" );

   /*#
     @method wait Selector
     @brief Wait for the semaphore to be  to be open.
     @optparam timeout Milliseconds to wait for the barrier to be open.
     @return true if the semaphore is signaled during the wait, false if the given timeout expires.

     If @b timeout is less than zero, the wait is endless; if @b timeout is zero,
     the wait exits immediately.
    */
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );
};

}


}

#endif	

/* end of semaphore.h */
