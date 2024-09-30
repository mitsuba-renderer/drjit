How to make a new release?
--------------------------

1. Ensure that the most recent version of Dr.Jit is checked out (including all
   submodules).

2. Ensure that the changelog is up to date in ``docs/changelog.rst``.

3. Verify that the CI is currently green all platforms.

4. Run the GHA "Build Python Wheels" with option "0". This effectively is a dry
   run of the wheel creation process.

5. If the action failed, fix whatever broke in the build process. If it succeded
   continue.

6. Update the version number in ``include/drjit/fwd.h``.

7. Add release number and date to ``docs/changelog.rst``.

8. Commit: ``git commit -am "vX.Y.Z release"``

9. Tag: ``git tag -a vX.Y.Z -m "vX.Y.Z release"``

10. Push: ``git push`` and ``git push --tags``

11. Run the GHA "Build Python Wheels" with option "1".

12. Check that the new version is available on
    `readthedocs <https://drjit.readthedocs.io/>`__.

13. Create a `release on GitHub <https://github.com/mitsuba-renderer/drjit/releases/new>`__
    from the tag created at step 10. The changelog can be copied from step 2.

