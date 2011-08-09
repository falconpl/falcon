/*
   FALCON - The Falcon Programming Language.
   FILE: modgroupbase.h

   Base abastract class for module groups and spaces.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 09 Aug 2011 00:43:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MODGROUPBASE_H_
#define _FALCON_MODGROUPBASE_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon 
{

class Module;

/** A simple class orderly guarding modules.
 */
class FALCON_DYN_CLASS ModGroupBase
{
public:
   typedef enum 
   {
      e_lm_load,
      e_lm_import_public,
      e_lm_import_private            
   }
   t_importMode;

   /** An entry of a the module map.
    An entry in a module map is composed of:
    - The pointer to the module;
    - The ownership status (whether the map can destroy the module or not);
    - The module load mode (load, import private or import public).
    */
      
   class Entry
   {
   public:
      
      Entry( Module* mod, t_importMode im, bool bOwn ):
         m_module(mod),
         m_imode(im),
         m_bOwn( bOwn )
      {}
      
      ~Entry();
      
      Module* module() const { return m_module; }
      bool own() const { return m_bOwn; }
      t_importMode imode() const { return m_imode; }

   private:
      Module* m_module;
      t_importMode m_imode;
      bool m_bOwn;
   };
  
   ModMap();
   ~ModMap();
   
   void add( Module* mod, t_importMode im, bool bown = true );
   void remove( Module* mod );
   
   Entry* findByURI( const String& path ) const;
   Entry* findByName( const String& name ) const;
   Entry* findByModule( Module* mod ) const;
   

   class EntryEnumerator 
   {
   public:
      virtual ~EntryEnumerator()=0;
      virtual void operator()( Entry* e );
   };
   
   void enumerate( EntryEnumerator& rator ) const;
   
private:
   class Private;
   Private* _p;
};

}

#endif	/* _FALCON_MODGROUPBASE_H_ */

/* end of modgroupbase.h */
