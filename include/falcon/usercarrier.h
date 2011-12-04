/*
   FALCON - The Falcon Programming Language.
   FILE: usercarrier.h

   Base class for ClassUser based instance reflection.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 07 Aug 2011 12:11:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_USERCARRIER_H_
#define _FALCON_USERCARRIER_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/item.h>

namespace Falcon {

class ClassUser;
class Item;

/** Class used to store unaware third party data in the Falcon GC.
 
 This base class can be used as a shell for external entities to be carried by
 an instance of a Class, generally a ClassUser.
 
 This class is meant to be derived by the final implementation which has
 to carry the entity coming from outside (i.e. from an host program). 
 
 The carrirer has support for basic GC marking and has an optional item
 vector which can be associated with the external data. The item vector is
 useful in case one or more of the properties associated with the external
 entity is a deep data (for instance, a string). In that case, creating
 a new garbage-sensible object each time a property is accessed is a waste;
 as such, it is possible to store the result of the property access in the
 item array, which will be automatically garbage marked when the carrier
 is marked.
 
 When the Falcon item goes out of scope, this carrier is destroyed by the GC.
 The subclass may do something sensible in the destructor, as i.e. destroying
 the carried data or communicate to the host program that the data is now
 free for use.
 
 Subclasses \b must provide a way to clone the data, or return 0 from the 
 clone() abstract virtual method if it is not possible to clone it. 
 */
class UserCarrier
{
public:
   UserCarrier();   
   UserCarrier( uint32  itemcount );   
   UserCarrier( const UserCarrier& other );
   
   virtual ~UserCarrier();
   virtual UserCarrier* clone() const { return new UserCarrier(*this); }  
   
   virtual void gcMark( uint32 mark );   
   uint32 gcMark() const { return m_gcMark; }

   Item* dataAt( uint32 pos ) const { return m_data + pos; }
   uint32 itemCount() const { return m_itemCount; }
   
private:
   
   Item* m_data;
   uint32 m_itemCount;
   uint32 m_gcMark;
};


template<class __T>
class UserCarrierT: public UserCarrier
{
public:
   UserCarrierT( __T* data ):
      m_data( data )
   {}
      
   UserCarrierT( __T* data, uint32  itemcount ):
      m_data( m_data )
   {}
   
   UserCarrierT( const UserCarrierT<__T>& other );

   __T* carried() const { return m_data; }
protected:
   virtual void* cloneData() const { return 0; } 

private:
   __T* m_data;
};

}

#endif	/* _FALCON_USERCARRIER_H_ */

/* end of usercarrier.h */
