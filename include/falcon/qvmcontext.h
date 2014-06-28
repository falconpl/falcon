/*
   FALCON - The Falcon Programming Language.
   FILE: QVMContext.h

   Falcon virtual machine -- queued VM context.
   -------------------------------------------------------------------
   Author: AlexRou

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_QVMCONTEXT_H_
#define FALCON_QVMCONTEXT_H_

#include <falcon/setup.h>
#include <falcon/vmcontext.h>
#include <falcon/mt.h>
#include <falcon/itemarray.h>

namespace Falcon
{

class TextReader;
class String;

class FALCON_DYN_CLASS QVMContext: public VMContext
{
	public:
		QVMContext( Process* prc, ContextGroup* grp = 0 );
		virtual ~QVMContext();

		void start( String name, Item item, int32 np = 0, Item const* params = 0, PStep* complete = 0, PStep* error = 0 );

		virtual void gcPerformMark();

        //Shouldn't be called
		virtual void onComplete();

		bool isSleeping()
		{
		    return m_isSleeping;
		}

		String runningItem()
		{
		    return m_running;
		}

	private:

	    class QueueItem;

	    typedef struct Node
	    {
	        QueueItem* data;
	        Node* next;
	    } Node;

        class PStepComplete;
        class PStepErrorGate;

		class PStepDeque;
		PStep* m_stepQueue;

		Function* m_baseFrame;

		Node* m_Queue;
		Node* m_QueueEnd;
		bool m_isSleeping;

		String m_running;
};

}

#endif /* FALCON_VMCONTEXT_H_ */

/* end of QVMContext.h */
