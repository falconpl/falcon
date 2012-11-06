/*
   FALCON - The Falcon Programming Language.
   FILE: mantra.cpp

   Basic "utterance" of the Falcon engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 25 Feb 2012 21:38:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)
*/

#undef SRC
#define SRC "engine/mantra.cpp"

#include <falcon/mantra.h>
#include <falcon/module.h>
#include <falcon/engine.h>

namespace Falcon
{

Mantra::Mantra():
   m_category( e_c_none ),
   m_module( 0 ),
   m_sr( 0, 0 ),
   m_mark(0)
{}

Mantra::Mantra( const String& name, int32 line, int32 chr ):
   m_category( e_c_none ),
   m_name( name ),
   m_module( 0 ),
   m_sr( line, chr ),
   m_mark(0)
{}

Mantra::Mantra( const String& name, Module* module, int32 line, int32 chr ):
   m_category( e_c_none ),
   m_name( name ),
   m_module( module ),
   m_sr( line, chr ),
   m_mark(0)
{}


Class* Mantra::handler() const
{
   static Class* cls = Engine::instance()->mantraClass();

   return cls;
}

void Mantra::locateTo( String& target ) const
{
   if ( m_module == 0 )
   {
      target = "<internal>:";
   }
   else {
      if( m_module->uri().size() != 0 )
      {
         target = m_module->uri();
      }
      else {
         target = "[" + m_module->name() + "]";
      }

      if( m_sr.line() != 0 )
      {
         target.A("(").N( m_sr.line() );
         if( m_sr.chr() )
         {
            target.A(":").N( m_sr.chr() );
         }
         target.A(")");
      }
      target += ":";
   }

   target += m_name;
}


void Mantra::gcMark( uint32 mark )
{
   if( mark > m_mark )
   {
      m_mark = mark;

      if( m_module != 0 ) {
         m_module->gcMark( mark );
      }
   }
}


bool Mantra::isCompatibleWith( Mantra::t_category cat ) const
{
   // are we searching anything?
   if( cat == Mantra::e_c_none )
   {
      return true;
   }
   // are we searching a class?
   else if( cat == Mantra::e_c_class )
   {
      if( this->category() == Mantra::e_c_class
          || this->category() == Mantra::e_c_falconclass
          || this->category() == Mantra::e_c_hyperclass
          || this->category() == Mantra::e_c_metaclass )
      {
         return true;
      }
   }
   // are we searching a function?
   else if( cat == Mantra::e_c_function )
   {
      if( this->category() == Mantra::e_c_function
          || this->category() == Mantra::e_c_pseudofunction
          || this->category() == Mantra::e_c_synfunction )
      {
         return true;
      }
   }
   // are we searching something more specific?
   else if( cat == this->category() )
   {
      return true;
   }

   return false;
}

}

/* end of mantra.cpp */

