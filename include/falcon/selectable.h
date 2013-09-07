/*
   FALCON - The Falcon Programming Language.
   FILE: selectable.h

   General interface for items that can be selected in a Selector
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Sep 2013 21:26:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_SELECTABLE_H_
#define _FALCON_SELECTABLE_H_

#include <falcon/setup.h>
#include <falcon/refcounter.h>
#include <falcon/class.h>
#include <falcon/multiplex.h>

namespace Falcon {



/* General interface for items that can be selected in a Selector.

 This is an interface exposed by those entities that want to be selected
 in a selector at script level.

 @note As this is an internal class, often used as a subclass in modules,
 it must not have the FALCON_DYN_SYM extension.

 @see Class::getSelectableInterface
 */
class Selectable
{
public:
   Selectable( const Class* cls, void* inst ):
      m_class(cls),
      m_instance(inst)
   {}

   const Class* handler() const { return m_class; }
   void* instance() const { return m_instance; }

   /** Returns the appropriate factory for the instance. */
   virtual const Multiplex::Factory* factory() const = 0;


   virtual void gcMark( uint32 mark )
   {
      m_class->gcMarkInstance( m_instance, mark );
   }

protected:
   virtual ~Selectable() {}

private:
   const Class* m_class;
   void* m_instance;

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(Selectable);
};

/**
 * Utility class used by multiplexers that know about the File Descriptor nature of the selectable classes.
 *
 * Some multiplexers can assume that all the selectable entities that are fed on that have an underlying
 * POSIX file descriptor somewhere in the selected instance.
 *
 * This pure virtual class is derived into concrete sublcasses that return the file descriptor held
 * by the Falcon instances they receive.
 */
class FDSelectable: public Selectable
{
public:
   FDSelectable( const Class* cls, void* inst ):
      Selectable( cls, inst )
   {}

   virtual ~FDSelectable() {};

   virtual int getFd() const = 0;
};
}

#endif /* SELECTABLE_H_ */

/* end of selectable.h */
