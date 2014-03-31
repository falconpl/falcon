/*
   FALCON - The Falcon Programming Language
   FILE: traceback.cpp

   Structure holding representation information of a point in code.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Mar 2014 18:14:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/traceback.cpp"

#include <falcon/traceback.h>
#include <deque>

namespace Falcon {

class TraceBack::Private
{
public:
   typedef std::deque<TraceStep*> StepList;
   StepList m_steps;

   Private() {}

   ~Private() {
      {
         StepList::iterator iter = m_steps.begin();
         while( iter != m_steps.end() )
         {
            TraceStep* step = *iter;
            delete step;
            ++iter;
         }
      }
   }


};

TraceBack::TraceBack()
{
   _p = new Private;
}


TraceBack::~TraceBack()
{
   delete _p;
}

void TraceBack::add(TraceStep* ts)
{
   _p->m_steps.push_back(ts);
}


String & TraceBack::toString( String &target, bool bAddPath, bool bAddParams ) const
{
   Private::StepList::const_iterator iter = _p->m_steps.begin();
   while( iter != _p->m_steps.end() )
   {
       target += "    ";
       const TraceStep& step = *(*iter);
       step.toString( target, bAddPath, bAddParams );
       ++iter;
       if(iter != _p->m_steps.end() )
       {
          target +="\n";
       }
   }

   return target;
}


length_t TraceBack::size() const
{
   return _p->m_steps.size();
}

TraceStep* TraceBack::at(length_t pos) const
{
   if( pos < _p->m_steps.size() )
   {
      return _p->m_steps[pos];
   }
   else
   {
      return 0;
   }
}


/** Enumerate the traceback steps.
 \param rator A StepEnumerator that is called back with each step in turn.
 */
void TraceBack::enumerateSteps( StepEnumerator &rator ) const
{
   Private::StepList::iterator iter = _p->m_steps.begin();
   while( iter != _p->m_steps.end() )
   {
      TraceStep* step = *iter;
      rator(*step);
      ++iter;
   }

}

}

/* end of traceback.cpp */
