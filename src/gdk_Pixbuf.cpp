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

    Gtk::MethodTab methods[] =
    {
    { "version",            &Pixbuf::version },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Pixbuf, meth->name, meth->cb );
}


Pixbuf::Pixbuf( const Falcon::CoreClass* gen, const GdkPixbuf* buf )
    :
    Gtk::CoreGObject( gen, (GObject*) buf )
{}


Falcon::CoreObject* Pixbuf::factory( const Falcon::CoreClass* gen, void* buf, bool )
{
    return new Pixbuf( gen, (GdkPixbuf*) buf );
}


/*#
    @class gtk.GdkPixbuf
    @brief Information that describes an image.
 */

/*#
    @method version GdkPixbuf
    @brief a static function returning the GdkPixbuf version.
    @return the full version of the gdk-pixbuf library as a string.

    This is the version currently in use by a running program.
 */
FALCON_FUNC Pixbuf::version( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    vm->retval( new String( GDK_PIXBUF_VERSION ) );
}



} // Gdk
} // Falcon
