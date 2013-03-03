/*
   FALCON - The Falcon Programming Language.
   FILE: contextgroup.h

   Group of context sharing the same parallel execution.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 05 Aug 2012 22:44:21 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_CONTEXTGROUP_H
#define _FALCON_CONTEXTGROUP_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/refcounter.h>

#include  <falcon/atomic.h>

namespace Falcon {

namespace Ext {
   class ClassParallel;
}

class VMachine;
class Shared;
class VMContext;
class ItemArray;
class Error;

/**
 Group of contexts sharing the same parallel execution context.

 A parallel execution is a set of context that are meant to be executed
 commonly, and return a set of values that are then handled back to the
 launcher context that is waiting for them to finish.

 If an uncaught error arrives at the surface of a parallel execution, all
 the contexts in the group are interrupted and the error is repeated in the
 parent context.

 The virtual machine is created with a special main group having just one
 context, which is the main context. If an error reaches the surface of this
 context, it is repeated to the VM and ultimately to the embedding application.

 Groups can define a maximum number of processors on which they should run.
 The group is made runnable when it has at least one runnable context waiting
 to be executed and when there are less than the assigned number of processors
 currently running contexts from this group.

 Processors pick the first runnable context from a runnable group and run it
 as required.
 */
class FALCON_DYN_CLASS ContextGroup {

public:
   const static uint32 ANY_PROCESSOR=0xFFFFFFFF;

   /** Creates the group with a number of processors.
    \param owner The virtual machine owning this group.
    \param parent The context that started this group, if any.
    \param processors Maximum number of processors that are allowed to run contexts
    from this group (0 means unlimited).
    */
   ContextGroup( VMachine* owner, VMContext* parent=0, uint32 processors=ANY_PROCESSOR );

   /**
    Returns the virtual machine to which this group belongs to.
    */
   VMachine* owner() const { return m_owner; }

   /**
    Returns the context that initiated this group, if any.
    \note The master context group has no parent.
    */
   VMContext* parent() const { return m_parent; }

   /** Returns the count of contexts that can be run simultaneously.
       \note This number is never changed once assigned at creation,
       or anyhow after the group is added to the virtual machine.
    */
   uint32 assignedProcessors() const { return m_processors; }

   /** Change the count of contexts that can be run simultaneously.
       \note This number is never changed once assigned at creation,
       or anyhow after the group is added to the virtual machine.
    */
   void assignedProcessors( uint32 processors ) {m_processors = processors;}

   /** Returns the count of contexts that are running in this group.
       \note this method is used within the single-thread context manager only.
    */
   uint32 runningContexts() const;

   /** Increments the counter of the terminating contexts.

    When all the contexts are terminated, the terminated() shared must be
    signaled by the caller.

    \return true if this was the last context.
    */
   bool onContextTerminated(VMContext* ctx);

   /** Adds a new context to the group.
    \param ctx The context to be added to this group.

    \note This method is not interlocked; it should be invoked just
    while the group is being prepared to be sent to a virtual machine.
    It is not meant to add new contexts to the group while it is being
    actively managed by a virtual machine.

    */
   void addContext( VMContext* ctx );

   /** Called by the virtual machine as soon as the group is added.
    This method readies all the contexts, eventually handling all the contexts
    that can be immediately processed to the virtual machine.

    If the number of the processors allocated for this group is less
    than the context
    */
   void readyAllContexts();

   /** Called back (by the context manager) when a context is swapped out from a processor.
    \param ctx The context that has reached an idle state in the manager.
    */
   VMContext* onContextIdle();

   /** Called back (by the manager) when the context is considered ready to run.
    * \param ctx The contex whose schedule says it's ready to run.
    * \return true if the context is allowed to actually run, false if it's put in the quiescent set.
    */
   bool onContextReady( VMContext* ctx );

   /** Returns the results of all the contexts in the group.
    */
   ItemArray* results() const;

   /**
    Shared resource signaled when all the contexts are terminated.

    When the shared is signaled, the caller must be sure that any
    reference to this group is invalidated as the waiter might
    use or destroy this instance of ContextGroup immediately.
    */
   Shared* terminated() const { return m_termEvent; }

   /**
    Indicates that this thread group is to be terminated with error.
    \note If more contexts terminate contemporary with an error, only the
    first notified error is reported upstream. Other running contexts are
    terminated as soon as possible, and ready context are never re-scheduled.

    \param error The error that will be reported upstream.

    \note This class adds a reference count to the error that is removed
    when the ContextGroup instance is destroyed.
    */
   void setError( Error* error );

   Error* error() const;

   VMContext* getContext(uint32 count);
   uint32 getContextCount();
private:
   class Private;
   Private* _p;

   VMachine* m_owner;
   VMContext* m_parent;
   Shared* m_termEvent;
   uint32 m_processors;
   atomic_int m_terminated;

   ContextGroup();
   virtual ~ContextGroup();

   void configure( VMachine* owner, VMContext* parent=0, uint32 processors=ANY_PROCESSOR );
   friend class Ext::ClassParallel;

   FALCON_REFERENCECOUNT_DECLARE_INCDEC( ContextGroup );
};

}

#endif

/* end of contextgroup.h */
