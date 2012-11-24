/*
   FALCON - The Falcon Programming Language.
   FILE: refcounter.h

   Common member offering reference counting to host classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 23 Jul 2011 15:57:26 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_REFCOUNTER_H_
#define FALCON_REFCOUNTER_H_

#include <falcon/mt.h>
#include <falcon/atomic.h>

namespace Falcon {

/** Common member offering reference counting to host classes.

 An instance of this class can be made a member of a class that needs
 reference counting (as errors and modules) to standardize their reference
 counting interface.

 For instance:
 \code
 class InNeedOfRC
 {
 public:

   ....

 private:
   // IMPORTANT: It should be in PRIVATE
   \/\* virtual or not \*\/ ~InNeedOfRC() {}
   FALCON_REFERENCECOUNT_DECLARE_INCDEC(InNeedOfRC)
 };

 // ... user code ...
 InNeedOfRC item;
 item.incref();
 ...
 ...
 item.decref();
 \endcode

*/

template<class __T>
class FALCON_DYN_CLASS RefCounter
{
public:
   /** Creates the reference counter.
    \param owner The instance that will be destroyed as reference count hits 0.
    \param initCount Initial reference count (usually 1).
    */
   RefCounter( int32 initCount=1 ):
      m_count( initCount )
   {}

   /** Increments the reference count by one. */
   void inc() const { atomicInc(m_count); }

   /** Decrements the reference count by one.
    \this method may destroy the owner of the reference counter (and this item with it).
    */
   void dec(__T* data) { if( atomicDec(m_count)<= 0) delete data; }

private:
   mutable atomic_int m_count;
};

#define FALCON_REFERENCECOUNT_DECLARE_INCDEC(clsname) \
   private:\
      RefCounter<clsname> m_refcounter_##clsname; \
   public:\
   void incref() { m_refcounter_##clsname.inc(); }\
   void decref() { m_refcounter_##clsname.dec(this); }\
   friend class RefCounter<clsname>;\
   private:

}

#endif /* FALCON_REFCOUNTER_H_ */

/* end of refcounter.h */
