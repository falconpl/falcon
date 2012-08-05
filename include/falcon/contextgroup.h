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

namespace Falcon {

class VMachine;
class Shared;
class VMContext;
class ItemArray;

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
   /** Creates the group with a number of processors.
    \param owner The virtual machine owning this group.
    \param parent The context that started this group, if any.
    \param processors Maximum number of processors that are allowed to run contexts
    from this group (0 means unlimited).
    */
   ContextGroup( VMachine* owner, VMContext* parent=0, int32 processors=0 );
   virtual ~ContextGroup();

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
       \note this method is used within the single-thread context manager only.
    */
   int32 assignedProcessors() const { return m_processors; }

   /** Change the count of contexts that can be run simultaneously.
       \note this method is used within the single-thread context manager only.
    */
   void assignedProcessors( int32 processors ) {m_processors = processors;}

   /** Returns the count of contexts that are running in this group.
       \note this method is used within the single-thread context manager only.
    */
   int32 runningContexts() const { return m_running; }

   /** Returns the count of contexts that are running in this group.
       \note this method is used within the single-thread context manager only.
    */
   void runningContexts(int32 v) const { m_running = v; }

   /** Increments the counter of the terminating contexts.
    \note this method is used within the single-thread context manager only.

    When all the contexts are terminated, the terminated() shared must be
    signaled by the caller.

    \return true if this was the last context.
    */
   bool terminateContext();
   void addContext( VMContext* ctx );

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
    Gets the next context ready to run in this group.
    \return The context ready to run or 0 if none.
    */
   VMContext* nextReadyContext();
   /**
    Puts a context in ready-to-run state.

    \note This method doesn't generate any signal to wake up processors that
    may want to run the readied context. It's the context manager that must
    add the group in the ready groups lists and then notify the processors.
    However, if a processor was already looking at this group to get a
    ready context, it is possible that the context get assigned before
    the signaling is performed.
    */
   void addReadyContext( VMContext* ctx );

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

private:
   class Private;
   Private* _p;

   VMachine* m_owner;
   VMContext* m_parent;
   Shared* m_termEvent;
   int32 m_processors;
   int32 m_running;
   int32 m_terminated;
};

}

#endif

/* end of contextgroup.h */
