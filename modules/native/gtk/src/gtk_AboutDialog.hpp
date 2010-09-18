#ifndef GTK_ABOUTDIALOG_HPP
#define GTK_ABOUTDIALOG_HPP

#include "modgtk.hpp"

#if GTK_CHECK_VERSION( 2, 6, 0 )

namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::AboutDialog
 */
class AboutDialog
    :
    public Gtk::CoreGObject
{
public:

    AboutDialog( const Falcon::CoreClass*, const GtkAboutDialog* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC get_name( VMARG );

    static FALCON_FUNC set_name( VMARG );

#if GTK_CHECK_VERSION( 2, 12, 0 )
    static FALCON_FUNC get_program_name( VMARG );

    static FALCON_FUNC set_program_name( VMARG );
#endif

    static FALCON_FUNC get_version( VMARG );

    static FALCON_FUNC set_version( VMARG );

    static FALCON_FUNC get_copyright( VMARG );

    static FALCON_FUNC set_copyright( VMARG );

    static FALCON_FUNC get_comments( VMARG );

    static FALCON_FUNC set_comments( VMARG );

    static FALCON_FUNC get_license( VMARG );

    static FALCON_FUNC set_license( VMARG );

#if GTK_CHECK_VERSION( 2, 8, 0 )
    static FALCON_FUNC get_wrap_license( VMARG );

    static FALCON_FUNC set_wrap_license( VMARG );
#endif

    static FALCON_FUNC get_website( VMARG );

    static FALCON_FUNC set_website( VMARG );

    static FALCON_FUNC get_website_label( VMARG );

    static FALCON_FUNC set_website_label( VMARG );

    static FALCON_FUNC get_authors( VMARG );

    static FALCON_FUNC set_authors( VMARG );

    static FALCON_FUNC get_artists( VMARG );

    static FALCON_FUNC set_artists( VMARG );

    static FALCON_FUNC get_documenters( VMARG );

    static FALCON_FUNC set_documenters( VMARG );

    static FALCON_FUNC get_translator_credits( VMARG );

    static FALCON_FUNC set_translator_credits( VMARG );

    static FALCON_FUNC get_logo( VMARG );

    static FALCON_FUNC set_logo( VMARG );

    static FALCON_FUNC get_logo_icon_name( VMARG );

    static FALCON_FUNC set_logo_icon_name( VMARG );

    static FALCON_FUNC set_email_hook( VMARG );

    static FALCON_FUNC set_url_hook( VMARG );

    //static FALCON_FUNC show_about_dialog( VMARG );

};


/*
 *  email hook stuff
 */
extern Falcon::GarbageLock*     about_dialog_email_hook_func_item;
extern Falcon::GarbageLock*     about_dialog_email_hook_data_item;
void about_dialog_email_hook_func( GtkAboutDialog*, const gchar*, gpointer );


/*
 *  url hook stuff
 */
extern Falcon::GarbageLock*     about_dialog_url_hook_func_item;
extern Falcon::GarbageLock*     about_dialog_url_hook_data_item;
void about_dialog_url_hook_func( GtkAboutDialog*, const gchar*, gpointer );


} // Gtk
} // Falcon

#endif // GTK_CHECK_VERSION( 2, 6, 0 )

#endif // !GTK_ABOUTDIALOG_HPP
