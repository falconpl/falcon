/*
   FALCON - The Falcon Programming Language.
   FILE: genericitem.h

   eneric item with standardized behavior.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 12 Aug 2011 21:06:49 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_GENERICITEM_H_
#define _FALCON_GENERICITEM_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

/** Generic item with standardized behavior.
 
 This simple class offers a base for items shared with the engine through 
 ClassGeneric. This is meant to be subclasses by instances with minimal 
 requirements. See ClassGeneric for more details.
 */
class FALCON_DYN_CLASS GenericItem
{
public:
   GenericItem( const String& name ):
      m_name( name )
   {}
   
   const String& name() const { return m_name; }
      
   virtual ~GenericItem();
   virtual void gcMark( uint32 value ) = 0;
   virtual bool gcCheck( uint32 value ) = 0;
   
   /** Return a copy of this generic item, or 0 if not allowed to copy. */
   virtual GenericItem* clone() const = 0;
   virtual void describe( String& target ) const = 0;
   
private:
   String m_name;
};

}

#endif	/* _FALCON_GENERICITEM_H_ */

/* end of genericitem.h */
