#ifndef GTK_MESSAGEDIALOG_HPP
#define GTK_MESSAGEDIALOG_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::MessageDialog
 */
class MessageDialog
    :
    public Gtk::CoreGObject
{
public:

    MessageDialog( const Falcon::CoreClass*, const GtkMessageDialog* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_with_markup( VMARG );

    static FALCON_FUNC set_markup( VMARG );

    static FALCON_FUNC set_image( VMARG );
#if GTK_CHECK_VERSION( 2, 14, 0 )
    static FALCON_FUNC get_image( VMARG );
#endif
    static FALCON_FUNC set_secondary_text( VMARG );

    static FALCON_FUNC set_secondary_markup( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_MESSAGEDIALOG_HPP
