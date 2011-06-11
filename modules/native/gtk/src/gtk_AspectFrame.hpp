#ifndef GTK_ASPECTFRAME_HPP
#define GTK_ASPECTFRAME_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::AspectFrame
 */
class AspectFrame
    :
    public Gtk::CoreGObject
{
public:

    AspectFrame( const Falcon::CoreClass*, const GtkAspectFrame* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC set( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_ASPECTFRAME_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
