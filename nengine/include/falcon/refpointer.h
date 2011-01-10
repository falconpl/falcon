/*
   FALCON - The Falcon Programming Language.
   FILE: refpointer.h

   Automatically manages reference-count oriented datatype.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 09 Jan 2011 13:37:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_REFPOINTER_H_
#define FALCON_REFPOINTER_H_


namespace Falcon {

/** Automatically manages reference-count oriented datatype.
 *
 * The template class must be applied to classes exposing
 * an incref() and decref() method.
 *
 * A Referenceable derived class has this ability, but the
 * template doesn't check if the class is derived from Referenceable;
 * just, it must expose the public methods incref and decref.
 *
 * @note: The initial reference count of a class subject to ref_ptr
 *        should be zero, as ref_ptr will call incref() upon receival.
 */

template<class __T>
class ref_ptr {
   ref_ptr() {
      data = 0;
   }

   ref_ptr( __T* data )
   {
      data->incref();
      m_data = data;
   }

   ref_ptr( const ref_ptr& other ) {
      m_data = other.m_data;
      m_data->incref();
   }

   ~ref_ptr() {
      if( m_data != 0 ) m_data->decref();
   }

   __T* operator->() { return m_data; }

private:
   __T* m_data;
};

}

#endif /* FALCON_REFPOINTER_H_ */
