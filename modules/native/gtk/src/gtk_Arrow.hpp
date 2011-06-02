#ifndef GTK_ARROW_HPP
#define GTK_ARROW_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Arrow
 */
class Arrow
    :
    public Gtk::CoreGObject
{
public:

    Arrow( const Falcon::CoreClass*, const GtkArrow* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_ARROW_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
