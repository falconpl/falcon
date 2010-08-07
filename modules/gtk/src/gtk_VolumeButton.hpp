#ifndef GTK_VOLUMEBUTTON_HPP
#define GTK_VOLUMEBUTTON_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::VolumeButton
 */
class VolumeButton
    :
    public Gtk::CoreGObject
{
public:

    VolumeButton( const Falcon::CoreClass*, const GtkVolumeButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_VOLUMEBUTTON_HPP
