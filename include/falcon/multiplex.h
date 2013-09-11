/*
   FALCON - The Falcon Programming Language.
   FILE: multiplex.h

   Multiplex framework for Streams and Selectors.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Feb 2013 18:18:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MULTIPLEX_H_
#define _FALCON_MULTIPLEX_H_

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon
{

class Selectable;
class Selector;

/**
 * Stream multiplexer.
 *
 * A class that can multiplex stream instances polling for their
 * availability status.
 *
 * This class is usually not created directly; the MultiplexGenerator::generate()
 * method is used to produce new instances of this class.
 *
 * \note The multiplex instances are owned by the selector where they
 * are instantiated.
 *
 * \see getMultiplexGenerator();
 */
class FALCON_DYN_CLASS Multiplex
{
public:
   virtual ~Multiplex(){}

   /** Adds a resource to the multiplexing.
    *
    * \param resource The resource to be added (or modified).
    * \param mode A read/write/error poll mode as bit-field according to Selector::mode_*
    *
    */
   virtual void add( Selectable* resource, int mode ) = 0;

   /** Removes a resource from multiplexing (as soon as possible).
    *
    * \param Stream stream to be removed.
    *
    * The subclass should remove the multiplexed entity as soon as possible,
    * but it is allowed to send this same stream to a ready queue in the
    * selector after this call. So, in case it's difficult or impractical to
    * remove this stream from the selection right now, this can be delayed to
    * a future moment.
    *
    * \param Management of refcounts for streams is up to the multiplex instance;
    * the selector has its own counts.
    */
   virtual void remove( Selectable* resource ) = 0 ;

   /** Returns the count of resources waited upon by this multiplex.
    It is important that this number is accurate, to allow selectors to
    properly unload the multiplex when all the resources are removed from it.
    */
   virtual uint32 size() const = 0;

   Selector* selector() const { return m_selector; }

   class FALCON_DYN_CLASS Factory
   {
   public:
      virtual ~Factory() {}
      virtual Multiplex* create( Selector* selector ) const = 0;
   };

   /** Return the factory that can be used to create new multiplex */
   const Factory* factory() const { return m_factory; }

protected:

   /** Creates the multiplex on a selector.
    * The module is used for back-reference and keep alive marks.
    */
   Multiplex( const Factory* factory, Selector* master ):
      m_selector(master),
      m_factory( factory )
   {}

   friend class Factory;

   /**
    * Invoked when a resource is ready for read.
    * \param stream The stream now ready for read.
    *
    * Call this back when a stream is found ready for read.
    * \note This function increments the reference count for the stream;
    * the count is decremented as the stream is dequeued by the Selector's user.
    */
   virtual void onReadyRead( Selectable* resource );

   /**
    * Invoked when a resource is ready for write.
    * \param stream The stream now ready for write.
    *
    * Call this back when a stream is found ready for write.
    * \note This function increments the reference count for the stream;
    * the count is decremented as the stream is dequeued by the Selector's user.
    */
   virtual void onReadyWrite( Selectable* resource );

   /**
    * Invoked when a resource is found having an error signal active.
    * \param stream The stream has an error signal active.
    *
    * Call this back when a stream line has an error, high priority
    * or out of band signal active.
    *
    * \note This function increments the reference count for the stream;
    * the count is decremented as the stream is dequeued by the Selector's user.
    */
   virtual void onReadyErr( Selectable* resource );

   Selector* m_selector;
   const Factory* m_factory;
};



}

#endif /* MULTIPLEX_H_ */

/* end of multiplex.h */
