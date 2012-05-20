/*
   FALCON - The Falcon Programming Language.
   FILE: derivedfrom.h

   Class implementing common behavior for classes with a single parent.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 20:54:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DERIVEDFROM_H_
#define _FALCON_DERIVEDFROM_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon {

class String;

/** Abstract class implementing commons for classes derived from a single parent.
 
 */
class DerivedFrom: public Class
{
public:
   DerivedFrom( Class* parent, const String& name );
   virtual ~DerivedFrom() {}

   /** A shortcut to get a property from the parent. */
   bool op_getParentProperty( VMContext* ctx, void* instance, const String& prop ) const;
   
   
   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   virtual bool isDerivedFrom( const Class* cls ) const;
   virtual Class* getParent( const String& name ) const;
   virtual void* getParentData( Class* parent, void* data ) const;
   virtual void enumerateParents( Class::ClassEnumerator& pe ) const;
   
   /** This just enumerates the parent's properties.
    
    Very probably you'll want to override this by adding your own 
    properties and then calling directing m_parent->enumerate*, however this
    method provides a sensible minimal behavior.
    */
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;

   /** This just enumerates the parent's properties.
    
    Very probably you'll want to override this by adding your own 
    properties and then calling directing m_parent->enumerate*, however this
    method provides a sensible minimal behavior.
    */
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;

   /** This just hooks into parent's hasProperty.
    
    Very probably you'll want to override this by adding your own 
    properties and then calling directing m_parent->hasProerty, however this
    method provides a sensible minimal behavior.
    */
   virtual bool hasProperty( void* instance, const String& prop ) const;

   /** This just hooks into parent's op_getProperty.
    
    Very probably you'll want to override this by adding your own 
    properties, however this method provides a sensible minimal behavior.
    */
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   
   /** This just hooks into parent's op_getProperty.
    
    Very probably you'll want to override this by adding your own 
    properties, however this method provides a sensible minimal behavior.
    */
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop) const;
   
      
   
   //=========================================
   // Instance management

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   //=========================================================
   // Class management
   //

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   //=========================================================
   // Operators.
   //
   
   virtual void op_neg( VMContext* ctx, void* instance ) const;
   virtual void op_add( VMContext* ctx, void* instance ) const;
   virtual void op_sub( VMContext* ctx, void* instance ) const;
   virtual void op_mul( VMContext* ctx, void* instance ) const;
   virtual void op_div( VMContext* ctx, void* instance ) const;
   virtual void op_mod( VMContext* ctx, void* instance ) const;
   virtual void op_pow( VMContext* ctx, void* instance ) const;
   virtual void op_shr( VMContext* ctx, void* instance ) const;
   virtual void op_shl( VMContext* ctx, void* instance ) const;   
   virtual void op_aadd( VMContext* ctx, void* instance) const;
   virtual void op_asub( VMContext* ctx, void* instance ) const;
   virtual void op_amul( VMContext* ctx, void* instance ) const;
   virtual void op_adiv( VMContext* ctx, void* instance ) const;
   virtual void op_amod( VMContext* ctx, void* instance ) const;
   virtual void op_apow( VMContext* ctx, void* instance ) const;
   virtual void op_ashr( VMContext* ctx, void* instance ) const;
   virtual void op_ashl( VMContext* ctx, void* instance ) const;
   virtual void op_inc( VMContext* vm, void* instance ) const;
   virtual void op_dec(VMContext* vm, void* instance) const;
   virtual void op_incpost(VMContext* vm, void* instance ) const;
   virtual void op_decpost(VMContext* vm, void* instance ) const;
   virtual void op_getIndex(VMContext* vm, void* instance ) const;
   virtual void op_setIndex(VMContext* vm, void* instance ) const;
   virtual void op_compare( VMContext* ctx, void* instance ) const;
   virtual void op_isTrue( VMContext* ctx, void* instance ) const;
   virtual void op_in( VMContext* ctx, void* instance ) const;   
   virtual void op_call( VMContext* ctx, int32 paramCount, void* instance ) const;
   virtual void op_eval( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* instance ) const;
   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;

protected:
   Class* m_parent;
};

}
#endif 

/* end of derivedfrom.h */
