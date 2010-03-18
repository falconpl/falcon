/**
 *  \file gdk_Pixbuf.cpp
 */

#include "gdk_Pixbuf.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void Pixbuf::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Pixbuf = mod->addClass( "GdkPixbuf", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_Pixbuf->getClassDef()->addInheritance( in );
#if 0
    Gtk::MethodTab methods[] =
    {
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Pixbuf, meth->name, meth->cb );
#endif
}


Pixbuf::Pixbuf( const Falcon::CoreClass* gen, const GdkPixbuf* buf )
    :
    Gtk::CoreGObject( gen )
{
    if ( buf )
        setUserData( new Gtk::GData( (GObject*) buf ) );
}


Falcon::CoreObject* Pixbuf::factory( const Falcon::CoreClass* gen, void* buf, bool )
{
    return new Pixbuf( gen, (GdkPixbuf*) buf );
}


/*#
    @class gtk.GdkPixbuf
    @brief Information that describes an image.
 */


} // Gdk
} // Falcon
